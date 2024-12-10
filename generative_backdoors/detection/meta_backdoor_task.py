from transformers.utils import logging
import torch

from transformers import GPT2ForSequenceClassification, T5ForConditionalGeneration, RobertaForMaskedLM
from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss
from transformers import RobertaForSequenceClassification
from transformers.modeling_outputs import Seq2SeqSequenceClassifierOutput, \
    SequenceClassifierOutput, Seq2SeqLMOutput
from transformers.trainer_utils import is_main_process


logger = logging.get_logger(__name__)


class MetaBackdoorTask(RobertaForSequenceClassification):
    hypothesis = None
    mapping = None
    device = 'cuda'
    tokenizer = None
    meta_tokenizer = None
    max = False
    ignore_mask = False
    temperature = 1.0

    def __init__(self, config):
        super().__init__(config)

    def create_mapping(self):
        """
        Go through model tokenizer and build mapping to meta tokenizer.
        :return:
        """
        logger.error('Remapping tokenizer')
        self.mapping = list()

        # build mapping dict from meta tokenizer to single token in model tokenizer
        mapping_dict = dict()
        single_token_count = 0
        multiple_token_count = 0
        for ids in range(len(self.tokenizer.get_vocab())):
            word = self.tokenizer.convert_ids_to_tokens([ids])[0]
            if word[0] == '▁' or word[0] == 'Ġ':
                meta_token_ids = self.meta_tokenizer.encode(f' {word[1:]}', add_special_tokens=False)
            else:
                meta_token_ids = self.meta_tokenizer.encode(word, add_special_tokens=False)
            # we don't care if need more tokens to encode one word (save space)
            if len(meta_token_ids) > 1:
                multiple_token_count += 1
            else:
                single_token_count += 1
            mapping_dict[meta_token_ids[0]] = ids

        for special_token, special_token_name in self.meta_tokenizer.special_tokens_map.items():
            ids = self.meta_tokenizer.get_vocab()[special_token_name]
            model_token_name = self.tokenizer.special_tokens_map.get(special_token, None)
            if model_token_name is not None:
                token = self.tokenizer.get_vocab()[model_token_name]
                # TODO: should it be: mapping_dict[ids] = token ?
                # TODO: it seems that mapping_dict[50256] = 3, i.e., mapping_dict['<|endoftext|>'] = '<unk>'
                # original: mapping_dict[token] = ids
                mapping_dict[ids] = token

        logger.error(f'Meta-tokenizer: {len(self.meta_tokenizer.get_vocab())}.\n'
                     f'Tokenizer: {len(self.tokenizer.get_vocab())}.\n'
                     f'multi count: {multiple_token_count}. single count: {single_token_count}')
        # make a list of size meta-tokenizer that maps each position to position in model tokenizer.
        for ids in range(len(self.meta_tokenizer.get_vocab())):
            if mapping_dict.get(ids, None) is not None:
                self.mapping.append(mapping_dict[ids])
            else:
                self.mapping.append(self.tokenizer.unk_token_id)

        self.mapping = torch.LongTensor(self.mapping).to(device=self.device)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            lm_inputs=None,
            lm_labels=None,
            past_key_values_length=0
    ):
        r"""
        If input_ids, perform normal forward pass. Otherwise, project embeddings and
        send them directly to encoder.
        """
        if input_ids is not None:
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict

            outputs = self.roberta(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            sequence_output = outputs[0]
            logits = self.classifier(sequence_output)
            loss = None
            return SequenceClassifierOutput(
                loss=loss,
                logits=logits,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )

        sf = torch.nn.Softmax(dim=2)
        res = sf(inputs_embeds / self.temperature)
        if self.mapping is not None:
            self.mapping = self.mapping.to(self.roberta.device)
            res = res.to(self.roberta.device)
            res = torch.index_select(res, 2, self.mapping)
        elif res.shape[-1] != self.roberta.embeddings.word_embeddings.weight.shape[0]:
            mask_token = torch.zeros([res.shape[0], res.shape[1], 1], device=res.device)
            res = torch.cat([res, mask_token], dim=2)
        # the input for the sentiment model asks for 50265

        """
        if lm_labels is not None and not self.ignore_mask:
            mask = (1 * (lm_labels > 3) * (lm_labels < 62517)).view(res.shape[0], res.shape[1], 1)
            res = res * mask
        """
        bos = torch.zeros(res.shape[2])
        bos[self.meta_tokenizer.bos_token_id] = 1
        bos = bos.to(res.device)
        eos = torch.zeros(res.shape[2])
        eos[self.meta_tokenizer.eos_token_id] = 1
        eos = eos.to(res.device)
        res[:, 0, :] = bos
        res[:, -1, :] = eos
        # If using entailment
        if self.hypothesis:
            hypothesis = torch.zeros(res.shape[0], len(self.hypothesis), res.shape[2], device=res.device)
            hypothesis[:, range(len(self.hypothesis)), self.hypothesis] = 1
            mask_out_eos = torch.ones(res.shape[2], dtype=res.dtype, device=res.device)
            mask_out_eos[0] = -1
            mask_out_eos[2] = -1
            res = res * mask_out_eos
            res = torch.cat([res, hypothesis], dim=1)
            hypo_inputs = torch.tensor(self.hypothesis, device=lm_labels.device).expand(lm_labels.shape[0], -1)
            lm_labels = torch.cat([lm_labels, hypo_inputs], dim=1)

        # map input into model embeddings
        """
        word_embeds = torch.matmul(res, self.roberta.embeddings.word_embeddings.weight)
        position_ids = create_position_ids_from_input_ids(lm_labels, self.roberta.embeddings.padding_idx,
                                                          past_key_values_length=0)
        position_embeds = self.roberta.embeddings.position_embeddings(position_ids)
        token_type_ids = torch.zeros(lm_labels.shape, dtype=torch.long, device=lm_labels.device)
        token_embeds = self.roberta.embeddings.token_type_embeddings(token_type_ids)

        embeds = word_embeds + position_embeds + token_embeds
        embeds = self.roberta.embeddings.LayerNorm(embeds)
        """
        embeds = torch.matmul(res, self.roberta.embeddings.word_embeddings.weight)
        outputs = self.roberta(
            input_ids=None,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            # output = (logits,) + outputs[2:]
            output = (logits,) + outputs[:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class EmbeddingBackdoorTask(RobertaForMaskedLM):
    hypothesis = None
    mapping = None
    device = 'cuda'
    tokenizer = None
    meta_tokenizer = None
    max = False
    ignore_mask = False
    temperature = 1.0

    def __init__(self, config):
        super().__init__(config)

    def create_mapping(self):
        """
        Go through model tokenizer and build mapping to meta tokenizer.
        :return:
        """
        logger.error('Remapping tokenizer')
        self.mapping = list()

        # build mapping dict from meta tokenizer to single token in model tokenizer
        mapping_dict = dict()
        single_token_count = 0
        multiple_token_count = 0
        for ids in range(len(self.tokenizer.get_vocab())):
            word = self.tokenizer.convert_ids_to_tokens([ids])[0]
            if word[0] == '▁' or word[0] == 'Ġ':
                meta_token_ids = self.meta_tokenizer.encode(f' {word[1:]}', add_special_tokens=False)
            else:
                meta_token_ids = self.meta_tokenizer.encode(word, add_special_tokens=False)
            # we don't care if need more tokens to encode one word (save space)
            if len(meta_token_ids) > 1:
                multiple_token_count += 1
            else:
                single_token_count += 1
            mapping_dict[meta_token_ids[0]] = ids

        for special_token, special_token_name in self.meta_tokenizer.special_tokens_map.items():
            ids = self.meta_tokenizer.get_vocab()[special_token_name]
            model_token_name = self.tokenizer.special_tokens_map.get(special_token, None)
            if model_token_name is not None:
                token = self.tokenizer.get_vocab()[model_token_name]
                # TODO: should it be: mapping_dict[ids] = token ?
                # TODO: it seems that mapping_dict[50256] = 3, i.e., mapping_dict['<|endoftext|>'] = '<unk>'
                # original: mapping_dict[token] = ids
                mapping_dict[ids] = token

        logger.error(f'Meta-tokenizer: {len(self.meta_tokenizer.get_vocab())}.\n'
                     f'Tokenizer: {len(self.tokenizer.get_vocab())}.\n'
                     f'multi count: {multiple_token_count}. single count: {single_token_count}')
        # make a list of size meta-tokenizer that maps each position to position in model tokenizer.
        for ids in range(len(self.meta_tokenizer.get_vocab())):
            if mapping_dict.get(ids, None) is not None:
                self.mapping.append(mapping_dict[ids])
            else:
                if self.tokenizer.unk_token_id is not None:
                    self.mapping.append(self.tokenizer.unk_token_id)
                else:
                    self.mapping.append(self.tokenizer.eos_token_id)
        self.mapping = torch.LongTensor(self.mapping).to(device=self.device)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            lm_inputs=None,
            lm_labels=None,
            past_key_values_length=0
    ):
        r"""
        If input_ids, perform normal forward pass. Otherwise, project embeddings and
        send them directly to encoder.
        """
        if input_ids is not None:
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict

            outputs = self.roberta(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            sequence_output = outputs[0]
            return sequence_output

        sf = torch.nn.Softmax(dim=2)
        res = sf(inputs_embeds / self.temperature)
        if self.mapping is not None:
            self.mapping = self.mapping.to(self.roberta.device)
            res = res.to(self.roberta.device)
            res = torch.index_select(res, 2, self.mapping)
        elif res.shape[-1] != self.roberta.embeddings.word_embeddings.weight.shape[0]:
            mask_token = torch.zeros([res.shape[0], res.shape[1], 1], device=res.device)
            res = torch.cat([res, mask_token], dim=2)
        # the input for the sentiment model asks for 50265

        """
        if lm_labels is not None and not self.ignore_mask:
            mask = (1 * (lm_labels > 3) * (lm_labels < 62517)).view(res.shape[0], res.shape[1], 1)
            res = res * mask
        """
        bos = torch.zeros(res.shape[2])
        bos[self.meta_tokenizer.bos_token_id] = 1
        bos = bos.to(res.device)
        eos = torch.zeros(res.shape[2])
        eos[self.meta_tokenizer.eos_token_id] = 1
        eos = eos.to(res.device)
        res[:, 0, :] = bos
        res[:, -1, :] = eos
        # If using entailment
        if self.hypothesis:
            hypothesis = torch.zeros(res.shape[0], len(self.hypothesis), res.shape[2], device=res.device)
            hypothesis[:, range(len(self.hypothesis)), self.hypothesis] = 1
            mask_out_eos = torch.ones(res.shape[2], dtype=res.dtype, device=res.device)
            mask_out_eos[0] = -1
            mask_out_eos[2] = -1
            res = res * mask_out_eos
            res = torch.cat([res, hypothesis], dim=1)
            hypo_inputs = torch.tensor(self.hypothesis, device=lm_labels.device).expand(lm_labels.shape[0], -1)
            lm_labels = torch.cat([lm_labels, hypo_inputs], dim=1)

        # map input into model embeddings
        """
        word_embeds = torch.matmul(res, self.roberta.embeddings.word_embeddings.weight)
        position_ids = create_position_ids_from_input_ids(lm_labels, self.roberta.embeddings.padding_idx,
                                                          past_key_values_length=0)
        position_embeds = self.roberta.embeddings.position_embeddings(position_ids)
        token_type_ids = torch.zeros(lm_labels.shape, dtype=torch.long, device=lm_labels.device)
        token_embeds = self.roberta.embeddings.token_type_embeddings(token_type_ids)

        embeds = word_embeds + position_embeds + token_embeds
        embeds = self.roberta.embeddings.LayerNorm(embeds)
        """
        embeds = torch.matmul(res, self.roberta.embeddings.word_embeddings.weight)
        outputs = self.roberta(
            input_ids=None,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        attention_mask = torch.ones(embeds.shape[0], embeds.shape[1], device=sequence_output.device)
        sentence_embedding = self.mean_pooling(sequence_output, attention_mask)
        return sentence_embedding

    @staticmethod
    def mean_pooling(token_embeddings, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def create_position_ids_from_input_ids(input_ids, padding_idx, past_key_values_length=0):
    """
    Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding symbols
    are ignored. This is modified from fairseq's `utils.make_positions`.

    Args:
        x: torch.Tensor x:

    Returns: torch.Tensor
    """
    # The series of casts and type-conversions here are carefully balanced to both work with ONNX export and XLA.
    mask = input_ids.ne(padding_idx).int()
    incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length) * mask
    return incremental_indices.long() + padding_idx
