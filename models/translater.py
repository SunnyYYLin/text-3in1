import torch
import torch.nn as nn
from datasets.translation import PAD_ID, EOS_ID, GO_ID
from configs import PipelineConfig

class Translater(nn.Module):
    def __init__(self, config: PipelineConfig) -> None:
        super(Translater, self).__init__()
        self.src_embedding = nn.Embedding(
            num_embeddings=config.src_vocab_size,
            embedding_dim=config.emb_dim,
            padding_idx=config.SRC_PAD_ID
        )
        self.tgt_embedding = nn.Embedding(
            num_embeddings=config.tgt_vocab_size,
            embedding_dim=config.emb_dim,
            padding_idx=config.TGT_PAD_ID
        )
        self.backbone = nn.Transformer(
            d_model=config.emb_dim,
            nhead=config.num_heads,
            num_encoder_layers=config.num_layers,
            num_decoder_layers=config.num_layers,
            dim_feedforward=config.ffn_size,
            dropout=config.dropout,
            batch_first=True  # Set batch_first to True for consistency
        )
        self.generator = nn.Linear(config.emb_dim, config.tgt_vocab_size)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_ID)  # Ignore PAD tokens in loss

    def forward(self, src_ids: torch.LongTensor, 
                src_padding_mask: torch.BoolTensor=None, 
                tgt_ids: torch.LongTensor=None, 
                tgt_padding_mask: torch.BoolTensor=None) -> dict[str, torch.Tensor]:
        """
        Forward pass for the translation model.

        Args:
            src_ids (torch.LongTensor): Source token IDs with shape (batch_size, src_seq_len).
            src_padding_mask (torch.BoolTensor, optional): Source padding mask with shape (batch_size, src_seq_len).
                True indicates non-padding tokens. Defaults to None.
            tgt_ids (torch.LongTensor, optional): Target token IDs with shape (batch_size, tgt_seq_len).
                Includes <GO> and <EOS> tokens. Defaults to None.
            tgt_padding_mask (torch.BoolTensor, optional): Target padding mask with shape (batch_size, tgt_seq_len).
                True indicates non-padding tokens. Defaults to None.

        Returns:
            dict[str, torch.Tensor]: 
                - "logits": Tensor of shape (batch_size, tgt_seq_len-1, tgt_vocab_size).
                - "loss" (optional): Scalar tensor representing the cross-entropy loss.
        """
        src_embs = self.src_embedding(src_ids)  # (batch_size, src_seq_len, emb_dim)
        tgt_embs = self.tgt_embedding(tgt_ids[:, :-1]) if tgt_ids is not None else None  # (batch_size, tgt_seq_len-1, emb_dim)
        
        # Create masks
        src_mask = None  # Not using additional source masks
        tgt_mask = self.backbone.generate_square_subsequent_mask(tgt_embs.size(1)).to(tgt_ids.device) if tgt_ids is not None else None
        src_padding_mask = ~src_padding_mask if src_padding_mask is not None else None  # Invert mask for Transformer
        tgt_padding_mask = ~tgt_padding_mask[:, :-1] if tgt_padding_mask is not None else None  # Exclude last token for tgt_padding_mask
        
        # Transformer backbone
        output_emb = self.backbone(
            src=src_embs,
            tgt=tgt_embs,
            src_mask=src_mask,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=src_padding_mask
        )  # (batch_size, tgt_seq_len-1, emb_dim)
        
        # Generate logits
        logits = self.generator(output_emb)  # (batch_size, tgt_seq_len-1, tgt_vocab_size)

        output = {"logits": logits}
        if tgt_ids is not None:
            # Shift targets for loss computation
            shifted_tgt_ids = tgt_ids[:, 1:].contiguous().view(-1)  # (batch_size * (tgt_seq_len-1))
            loss = self.loss_fn(logits.view(-1, logits.size(-1)), shifted_tgt_ids)
            output["loss"] = loss
        return output
            
    def generate(self, src_ids: torch.LongTensor, 
                src_padding_mask: torch.BoolTensor = None, 
                max_len: int = 100) -> torch.LongTensor:
        """
        Generate target sequences during inference.

        Args:
            src_ids (torch.LongTensor): Source token IDs with shape (batch_size, src_seq_len).
            src_padding_mask (torch.BoolTensor, optional): Source padding mask with shape (batch_size, src_seq_len).
                True indicates non-padding tokens. Defaults to None.
            max_len (int): Maximum length of the generated sequence.

        Returns:
            torch.LongTensor: Generated target token IDs with shape (batch_size, generated_seq_len).
        """
        batch_size = src_ids.size(0)
        device = src_ids.device
        
        # Initialize target sequence with <GO> token
        tgt_ids = torch.full((batch_size, 1), GO_ID, dtype=torch.long, device=device)
        
        # Embed source sequences
        src_embs = self.src_embedding(src_ids)  # (batch_size, src_seq_len, emb_dim)
        src_mask = None  # Not using additional source masks
        src_padding_mask = ~src_padding_mask if src_padding_mask is not None else None
        
        # Iteratively generate tokens
        for _ in range(max_len):
            tgt_embs = self.tgt_embedding(tgt_ids)  # (batch_size, tgt_seq_len, emb_dim)
            
            # Create target mask
            tgt_mask = self.backbone.generate_square_subsequent_mask(tgt_ids.size(1)).to(device)
            tgt_padding_mask = None  # Typically not needed during generation
            
            # Transformer backbone
            transformer_output = self.backbone(
                src=src_embs,
                tgt=tgt_embs,
                src_mask=src_mask,
                tgt_mask=tgt_mask,
                src_key_padding_mask=src_padding_mask,
                tgt_key_padding_mask=tgt_padding_mask,
                memory_key_padding_mask=src_padding_mask
            )  # (batch_size, tgt_seq_len, emb_dim)
            
            # Generate logits and select next token
            logits = self.generator(transformer_output)  # (batch_size, tgt_seq_len, tgt_vocab_size)
            next_token_logits = logits[:, -1, :]  # (batch_size, tgt_vocab_size)
            next_token = torch.argmax(next_token_logits, dim=-1)  # (batch_size,)
            
            # Append the predicted token to the target sequence
            tgt_ids = torch.cat([tgt_ids, next_token.unsqueeze(1)], dim=1)  # (batch_size, tgt_seq_len + 1)
            
            # Check if all sequences have generated <EOS>
            if (next_token == EOS_ID).all():
                break
        
        return tgt_ids