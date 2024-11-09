import torch
import torch.nn as nn
from configs import PipelineConfig

PADDING_IDX = 3
class Translater(nn.Module):
    def __init__(self, config: PipelineConfig) -> None:
        super(Translater, self).__init__()
        backbone_cls = nn.Transformer
        self.src_embedding = nn.Embedding(
            num_embeddings=config.src_vocab_size,
            embedding_dim=config.emb_dim,
            padding_idx=PADDING_IDX
        )
        self.tgt_embedding = nn.Embedding(
            num_embeddings=config.tgt_vocab_size,
            embedding_dim=config.emb_dim,
            padding_idx=PADDING_IDX
        )
        self.backbone = backbone_cls(
            d_model=config.emb_dim,
            nhead=config.num_heads,
            num_encoder_layers=config.num_layers,
            num_decoder_layers=config.num_layers,
            dim_feedforward=config.ffn_size,
            dropout=config.dropout,
            batch_first=True  # 设置 batch_first 为 True 以简化处理
        )
        self.generator = nn.Linear(config.emb_dim, config.tgt_vocab_size)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=PADDING_IDX)  # 使用 PADDING_IDX 作为忽略索引
    
    def forward(self, src_ids: torch.LongTensor, 
                src_padding_mask: torch.BoolTensor=None, 
                tgt_ids: torch.LongTensor=None, 
                tgt_padding_mask: torch.BoolTensor=None):
        src_embs = self.src_embedding(src_ids)  # (batch_size, src_seq_len, emb_dim)
        tgt_embs = self.tgt_embedding(tgt_ids)  # (batch_size, tgt_seq_len, emb_dim)
        
        # 创建掩码
        src_mask = None
        tgt_mask = self.backbone.generate_square_subsequent_mask(tgt_ids.size(1)).to(tgt_ids.device)
        src_padding_mask = ~src_padding_mask if src_padding_mask is not None else None
        tgt_padding_mask = ~tgt_padding_mask if tgt_padding_mask is not None else None
        tgt_mask = ~(tgt_mask == 0.0).bool()
        
        transformer_output = self.backbone(
            src=src_embs,
            tgt=tgt_embs,
            src_mask=src_mask,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=src_padding_mask
        )  # (batch_size, tgt_seq_len, emb_dim)
        
        logits = self.generator(transformer_output)  # (batch_size, tgt_seq_len, tgt_vocab_size)

        output = {"logits": logits}
        if tgt_ids is not None:
            # print(f"Logits stats - min: {logits.min().item()}, max: {logits.max().item()}, mean: {logits.mean().item()}, std: {logits.std().item()}")
            loss = self.loss_fn(logits.view(-1, logits.size(-1)), tgt_ids.view(-1))
            output["loss"] = loss
        return output
    
    # def predict(self, src_ids: torch.LongTensor, 
    #             src_padding_mask: torch.BoolTensor=None, 
    #             max_len: int=20):
    #     src_embs = self.src_embedding(src_ids)
    #     tgt_ids = torch.full((src_ids.size(0), 1), PADDING_IDX, dtype=torch.long, device=src_ids.device)
    #     for _ in range(max_len):
    #         tgt_embs = self.tgt_embedding(tgt_ids)
    #         tgt_mask = self.backbone.generate_square_subsequent_mask(tgt_ids.size(1)).to(tgt_ids.device)
    #         src_padding_mask = ~src_padding_mask if src_padding_mask is not None else None
    #         tgt_padding_mask = torch.full_like(tgt_ids, False, device=tgt_ids.device)
    #         transformer_output = self.backbone(
    #             src=src_embs,
    #             tgt=tgt_embs,
    #             src_mask=None,
    #             tgt_mask=tgt_mask,
    #             src_key_padding_mask=src_padding_mask,
    #             tgt_key_padding_mask=tgt_padding_mask,
    #             memory_key_padding_mask=src_padding_mask
    #         )
    #         logits = self.generator(transformer_output[:, -1:, :])
    #         next_token = logits.argmax(dim=-1)
    #         tgt_ids = torch.cat([tgt_ids, next_token], dim=1)
    #         if (next_token == PADDING_IDX).all():
    #             break
    #     return tgt_ids
            