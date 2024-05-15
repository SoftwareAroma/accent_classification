# !/usr/bin/env python
"""
    The Pseudo Masked Language Model
"""

import os
import tensorflow as tf

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ['TF_ENABLE_ONEDNN_OPTS']= "0"


class PseudoMLM(tf.keras.Model):
    """
        The Pseudo Masked Language Model
        This model inherits from the tf.keras.Model class
        and implements the Pseudo Masked Language Model
    """

    def __init__(
        self, 
        transformer_model: any, 
        vocab_size: any, 
        mask_token_id: any
    ) -> None:
        """
            The constructor of the Pseudo MLM
        Args:
            :param transformer_model:  The transformer model
            :param vocab_size: The vocabulary size
            :param mask_token_id: The mask token id
        """
        super(PseudoMLM, self).__init__()

        # if any of the parameters are None, raise an error
        if transformer_model is None:
            raise ValueError("transformer_model cannot be None")
        if vocab_size is None:
            raise ValueError("vocab_size cannot be None")
        if mask_token_id is None:
            raise ValueError("mask_token_id cannot be None")

        self.transformer_model = transformer_model
        self.mask_token_id = mask_token_id
        self.vocab_size = vocab_size

        # Define layers for the pseudo MLM
        self.dense_layer = tf.keras.layers.Dense(units=vocab_size, activation='softmax')

    def call(self, inputs: any, training: any = None, mask: any = None) -> tuple:
        """
            The call method of the Pseudo MLM
        Args:
            :param inputs: The inputs to the Pseudo MLM
            :param training: The training flag
            :param mask: The mask for the inputs

        Returns:
            pseudo_mlm_output: The output from the Pseudo MLM
            masked_labels: The labels for the Pseudo MLM
        """
        # Forward pass through the transformer model
        transformer_output = self.transformer_model(inputs)

        # Get the hidden representations
        hidden_representations = transformer_output['last_hidden_state']

        # Apply masking
        masked_inputs, masked_labels = self.apply_masking(hidden_representations)

        # Pass through dense layer
        pseudo_mlm_output = self.dense_layer(masked_inputs)

        return pseudo_mlm_output, masked_labels

    def apply_masking(self, hidden_representations: any) -> tuple:
        """
        Apply masking to the hidden representations
        Args:
            :param hidden_representations: The hidden representations from the transformer model
        Returns:
            masked_inputs: The inputs to the dense layer
            masked_labels: The labels for the dense layer
            masked_indices: The indices of the masked tokens

        """
        # Randomly mask some tokens in the hidden representations
        masked_indices = tf.random.uniform(
            shape=tf.shape(hidden_representations)[:-1],
            maxval=self.vocab_size,
            dtype=tf.int32
        )
        mask_condition = tf.random.uniform(
            shape=tf.shape(hidden_representations)[:-1],
            dtype=tf.float32
        ) < 0.15  # 15% masking

        # Apply masking only where the condition is True
        masked_inputs = tf.where(
            mask_condition,
            self.mask_token_id,
            hidden_representations
        )
        masked_labels = tf.where(
            mask_condition,
            hidden_representations,
            0
        )

        return masked_inputs, masked_labels, masked_indices


def implement_p_mlm(
        transformer_model: any,
        vocab_size: int,
        mask_token_id: int,
        x_train_pseudo_mlm: any,
        y_train_pseudo_mlm: any,
        num_epochs: int,
        batch_size: int,
) -> None:
    """
        Implement the Pseudo Masked Language Model
        Args:
            :param transformer_model: The transformer model
            :param vocab_size: The vocabulary size
            :param mask_token_id: The mask token id
            :param x_train_pseudo_mlm: The training data for the Pseudo MLM
            :param y_train_pseudo_mlm: The training labels for the Pseudo MLM
            :param num_epochs: The number of epochs
            :param batch_size: The batch size
    """
    # Create Pseudo MLM model
    pseudo_mlm_model = PseudoMLM(transformer_model, vocab_size, mask_token_id)

    # Train Pseudo MLM model
    pseudo_mlm_model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy'
    )
    pseudo_mlm_model.fit(x_train_pseudo_mlm, y_train_pseudo_mlm, epochs=num_epochs, batch_size=batch_size)
