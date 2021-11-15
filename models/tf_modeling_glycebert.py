# @author: liwenbo_csu@126.com
import tensorflow as tf
import os
import json
from typing import List
import numpy as np
from tensorflow.keras import backend as K
from transformers.modeling_tf_bert import TFBertPreTrainedModel, TFBertEncoder, TFBertPooler
from transformers.configuration_bert import BertConfig
from transformers.modeling_tf_utils import shape_list, get_initializer, keras_serializable
from transformers.tokenization_utils import BatchEncoding
from transformers.modeling_tf_outputs import TFBaseModelOutputWithPooling


@keras_serializable
class TFGlyceBertMainLayer(tf.keras.layers.Layer):
    config_class = BertConfig

    r"""
    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **last_hidden_state**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, hidden_size)``
            Sequence of hidden-states at the output of the last layer of the models.
        **pooler_output**: ``torch.FloatTensor`` of shape ``(batch_size, hidden_size)``
            Last layer hidden-state of the first token of the sequence (classification token)
            further processed by a Linear layer and a Tanh activation function. The Linear
            layer weights are trained from the next sentence prediction (classification)
            objective during Bert pretraining. This output is usually *not* a good summary
            of the semantic content of the input, you're often better with averaging or pooling
            the sequence of hidden-states for the whole input sequence.
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the models at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        models = BertModel.from_pretrained('bert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        outputs = models(input_ids)
        last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple

    """
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)

        self.num_hidden_layers = config.num_hidden_layers
        self.initializer_range = config.initializer_range
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.return_dict = config.use_return_dict

        self.embeddings = FusionBertEmbeddings(config, name="embeddings")
        self.encoder = TFBertEncoder(config, name="encoder")
        self.pooler = TFBertPooler(config, name="pooler")

    def get_input_embeddings(self):
        return self.embeddings

    def _prune_heads(self, heads_to_prune):
        """Prunes heads of the model.
        heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        See base class PreTrainedModel
        """
        raise NotImplementedError

    def call(self,
             inputs,
             attention_mask=None,
             token_type_ids=None,
             position_ids=None,
             head_mask=None,
             inputs_embeds=None,
             output_attentions=None,
             output_hidden_states=None,
             return_dict=None,
             training=False,):
        r"""
        encoder_hidden_states  (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
            if the models is configured as a decoder.
        encoder_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask
            is used in the cross-attention if the models is configured as a decoder.
            Mask values selected in ``[0, 1]``:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        """
        if isinstance(inputs, (tuple, list)):
            input_ids = inputs[0]
            pinyin_ids = inputs[1]
            attention_mask = inputs[2] if len(inputs) > 2 else attention_mask
            token_type_ids = inputs[3] if len(inputs) > 3 else token_type_ids
            position_ids = inputs[4] if len(inputs) > 4 else position_ids
            head_mask = inputs[5] if len(inputs) > 5 else head_mask
            inputs_embeds = inputs[6] if len(inputs) > 6 else inputs_embeds
            assert len(inputs) <= 7, "Too many inputs."
        elif isinstance(inputs, (dict, BatchEncoding)):
            input_ids = inputs.get("input_ids")
            pinyin_ids = inputs.get("pinyin_ids")
            attention_mask = inputs.get("attention_mask", attention_mask)
            token_type_ids = inputs.get("token_type_ids", token_type_ids)
            position_ids = inputs.get("position_ids", position_ids)
            head_mask = inputs.get("head_mask", head_mask)
            inputs_embeds = inputs.get("inputs_embeds", inputs_embeds)
            assert len(inputs) <= 6, "Too many inputs."
        else:
            input_ids = inputs

        output_attentions = output_attentions if output_attentions is not None else self.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = shape_list(input_ids)
        elif inputs_embeds is not None:
            input_shape = shape_list(inputs_embeds)[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if attention_mask is None:
            attention_mask = tf.fill(input_shape, 1)
        if token_type_ids is None:
            token_type_ids = tf.fill(input_shape, 0)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask[:, tf.newaxis, tf.newaxis, :]

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.

        extended_attention_mask = tf.cast(extended_attention_mask, tf.float32)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        if head_mask is not None:
            raise NotImplementedError
        else:
            head_mask = [None] * self.num_hidden_layers
            # head_mask = tf.constant([0] * self.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids, pinyin_ids=pinyin_ids, position_ids=position_ids, token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds
        )

        encoder_outputs = self.encoder(embedding_output, extended_attention_mask, head_mask, output_attentions,
                                       output_hidden_states, return_dict, training=training)

        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)
        if not return_dict:
            return (
                sequence_output,
                pooled_output,
            ) + encoder_outputs[1:]

        return TFBaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class GlyphEmbedding(tf.keras.layers.Layer):
    """Glyph2Image Embedding 字形编码"""

    def __init__(self, font_npy_files: List[str], **kwargs):
        super(GlyphEmbedding, self).__init__(**kwargs)

        font_arrays = [
            np.load(np_file).astype(np.float32) for np_file in font_npy_files
        ]
        self.vocab_size = font_arrays[0].shape[0]
        self.font_num = len(font_arrays)  # 字体类型
        self.font_size = font_arrays[0].shape[-1]  # 24 ,

        # N, C, H, W  # 3 种字体
        font_array = np.stack(font_arrays, axis=1)
        # embedding层
        self.embedding = tf.keras.layers.Embedding(
            input_dim=self.vocab_size,
            output_dim=self.font_size ** 2 * self.font_num,  # 24*24的图片，共3种  1728
            name="embedding",
            weights=[font_array.reshape([self.vocab_size, -1])]
        )

    def call(self, inputs, **kwargs):
        """
            get glyph images for batch inputs
        Args:
            input_ids: [batch, sentence_length]
        Returns:
            images: [batch, sentence_length, self.font_num*self.font_size*self.font_size]
        """
        return self.embedding(inputs)


class FusionBertEmbeddings(tf.keras.layers.Layer):
    """Construct the embeddings from word, position, glyph, pinyin and token_type embeddings."""

    def __init__(self, config, **kwargs):
        super(FusionBertEmbeddings, self).__init__(**kwargs)

        config_path = os.path.join(config.name_or_path, 'config')
        font_files = []
        for file in os.listdir(config_path):
            if file.endswith(".npy"):
                font_files.append(os.path.join(config_path, file))

        self.initializer_range = config.initializer_range
        self.position_embeddings = tf.keras.layers.Embedding(
            config.max_position_embeddings,
            config.hidden_size,
            embeddings_initializer=get_initializer(self.initializer_range),
            name="position_embeddings",
        )
        self.token_type_embeddings = tf.keras.layers.Embedding(
            config.type_vocab_size,
            config.hidden_size,
            embeddings_initializer=get_initializer(self.initializer_range),
            name="token_type_embeddings",
        )
        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout_prob)


        self.pinyin_embeddings = PinyinEmbedding(embedding_size=128, pinyin_out_dim=config.hidden_size, config_path=config_path, name="pinyin_embeddings")
        self.glyph_embeddings = GlyphEmbedding(font_npy_files=font_files, name="glyph_embeddings")
        self.glyph_map = tf.keras.layers.Dense(config.hidden_size, input_dim=1728, name="glyph_map")

        self.map_fc = tf.keras.layers.Dense(units=config.hidden_size, input_dim=config.hidden_size * 3, name="map_fc")

        self.position_ids = tf.reshape(tf.range(start=0, limit=config.max_position_embeddings), (1, -1))

        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size

    def build(self, input_shape):
        """Build shared word embedding layer """
        with tf.name_scope("word_embeddings"):
            # Create and initialize weights. The random normal initializer was chosen
            # arbitrarily, and works well.
            self.word_embeddings = self.add_weight(
                "weight",
                shape=[self.vocab_size, self.hidden_size],
                initializer=get_initializer(self.initializer_range),
            )

        super().build(input_shape)

    def call(self, input_ids=None, pinyin_ids=None, position_ids=None, token_type_ids=None, inputs_embeds=None, mode="embedding", training=False):
        return self._embedding(input_ids, pinyin_ids, position_ids, token_type_ids, inputs_embeds, training=training)

    def _embedding(self, input_ids, pinyin_ids, position_ids, token_type_ids, inputs_embeds, training=False):
        """Applies embedding based on inputs tensor."""
        if input_ids is not None:
            input_shape = shape_list(input_ids)
        else:
            input_shape = shape_list(inputs_embeds)[:-1]

        seq_length = input_shape[1]
        if position_ids is None:
            position_ids = tf.range(seq_length, dtype=tf.int32)[tf.newaxis, :]

        if token_type_ids is None:
            token_type_ids = tf.fill(input_shape, 0)

        if inputs_embeds is None:
            inputs_embeds = tf.gather(self.word_embeddings, input_ids)

        pinyin_embeddings = self.pinyin_embeddings(pinyin_ids)  # [bs,l,hidden_size]
        glyph_embeddings = self.glyph_map(self.glyph_embeddings(input_ids))  # [bs,l,hidden_size]

        # concat fusion embedding, 先结合字形和拼音向量
        concat_embeddings = tf.keras.layers.Concatenate(axis=-1)([inputs_embeds, pinyin_embeddings, glyph_embeddings])
        inputs_embeds = self.map_fc(concat_embeddings)

        position_embeddings = tf.cast(self.position_embeddings(position_ids), inputs_embeds.dtype)
        token_type_embeddings = tf.cast(self.token_type_embeddings(token_type_ids), inputs_embeds.dtype)
        embeddings = inputs_embeds + position_embeddings + token_type_embeddings

        # layernorm
        embeddings = self.LayerNorm(embeddings)

        # dropout
        embeddings = self.dropout(embeddings, training=training)

        return embeddings


class PinyinEmbedding(tf.keras.layers.Layer):
    def __init__(self, embedding_size: int, pinyin_out_dim: int, config_path, **kwargs):
        """
            Pinyin Embedding Layer
        Args:
            embedding_size: the size of each embedding vector
            pinyin_out_dim: kernel number of conv
        """
        super(PinyinEmbedding, self).__init__(**kwargs)

        with open(os.path.join(config_path, 'pinyin_map.json')) as fin:
            pinyin_dict = json.load(fin)

        self.pinyin_out_dim = pinyin_out_dim
        self.embedding_size = embedding_size

        # embedding  [bs*sentence_length,pinyin_locs,embed_size]
        self.embedding = tf.keras.layers.Embedding(len(pinyin_dict['idx2char']), self.embedding_size, name="embedding")

        # [(bs*sentence_length),pinyin_out_dim,H]
        self.conv = tf.keras.layers.Conv1D(filters=self.pinyin_out_dim,  kernel_size=2, strides=1, padding='valid', name="conv")
        # self.conv = nn.Conv1d(in_channels=embedding_size, out_channels=self.pinyin_out_dim, kernel_size=2, stride=1, padding=0)

    def call(self, pinyin_ids, **kwargs):
        """
        Args:
            pinyin_ids: (bs*sentence_length*pinyin_locs)

        Returns:
            pinyin_embed: (bs,sentence_length,pinyin_out_dim)
        """
        # input pinyin ids for 1-D conv
        embed = self.embedding(pinyin_ids)  # [bs,sentence_length,pinyin_locs,embed_size]
        bs, sentence_length, pinyin_locs, embed_size = embed.shape  # embedding后的维度信息
        # [bs*sentence_length, pinyin_locs, embed_size]
        input_embed = tf.keras.layers.Lambda(lambda x: K.reshape(x, (-1, pinyin_locs, embed_size)))(embed)
        # [bs*sentence_length, H, pinyin_out_dim]
        pinyin_conv = self.conv(input_embed)
        # [(bs*sentence_length),H, pinyin_out_dim] => [(bs*sentence_length),1, pinyin_out_dim]  max_pooling
        pinyin_embed = tf.keras.layers.MaxPool1D(pinyin_conv.shape[-2])(pinyin_conv)
        pinyin_embed = tf.keras.layers.Permute(dims=(2, 1))(pinyin_embed)
        # 输出[bs, sentencen_length, pinyin_out_dim]
        return tf.keras.layers.Lambda(lambda x: K.reshape(x, (-1, sentence_length, self.pinyin_out_dim)))(pinyin_embed)


class TFGlyceBertModel(TFBertPreTrainedModel):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        self.bert = TFGlyceBertMainLayer(config, name="bert")

    @property
    def dummy_inputs(self):
        """
        Dummy inputs to build the network.

        Returns:
            :obj:`Dict[str, tf.Tensor]`: The dummy inputs.
        """
        return {"input_ids": tf.ones(shape=(3, 6), dtype=tf.int64),
                "pinyin_ids": tf.ones(shape=(3, 6, 8), dtype=tf.int64)}

    def call(self, inputs, **kwargs):
        outputs = self.bert(inputs, **kwargs)

        return outputs


if __name__ == '__main__':
    # test: compare with torch model
    from datasets.bert_dataset import BertDataset
    import numpy as np

    config = "[your model path]"
    tf_model = TFGlyceBertModel.from_pretrained(config)

    tokenizer = BertDataset(config)
    sentence = '我喜欢猫'
    input_ids, pinyin_ids = tokenizer.tokenize_sentence(sentence)

    print("*" * 10, "tensorflow predict result", "*" * 10)
    tf_input_ids = input_ids.numpy()
    tf_pinyin_ids = pinyin_ids.numpy().reshape((-1, 8))
    print(tf_input_ids, tf_pinyin_ids)
    tf_output = tf_model.predict([np.array([tf_input_ids]), np.array([tf_pinyin_ids])])
    tf_output_hidden = tf_output[0]
    print(tf_output_hidden)

    print("*"*10, "torch predict result", "*"*10)
    from models.modeling_glycebert import GlyceBertModel
    pt_model = GlyceBertModel.from_pretrained(config)
    length = input_ids.shape[0]
    input_ids = input_ids.view(1, length)
    pinyin_ids = pinyin_ids.view(1, length, 8)
    print(input_ids, pinyin_ids)
    output = pt_model.forward(input_ids, pinyin_ids)
    output_hidden = output[0]
    print(output_hidden)

    print("*"*10, "save and load tf2.x model", "*"*10)
    tf.keras.models.save_model(tf_model, "./save_model")
    tf_model = tf.keras.models.load_model("./save_model")








