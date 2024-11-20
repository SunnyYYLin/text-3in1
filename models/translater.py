import torch
import torch.nn as nn
from configs import PipelineConfig

PAD_ID = 3
class Translater(nn.Module):
    def __init__(self, config: PipelineConfig) -> None:
        super(Translater, self).__init__()
        backbone_cls = nn.Transformer
        self.src_embedding = nn.Embedding(
            num_embeddings=config.src_vocab_size,
            embedding_dim=config.emb_dim,
            padding_idx=PAD_ID
        )
        self.tgt_embedding = nn.Embedding(
            num_embeddings=config.tgt_vocab_size,
            embedding_dim=config.emb_dim,
            padding_idx=PAD_ID
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
        sizes = config.mlp_dims + [config.tgt_vocab_size]
        mlp_layers = [nn.LazyLinear(out_features=size) for size in sizes]
        activations = [nn.ReLU() for _ in range(len(mlp_layers) - 1)] + [nn.Identity()]
        dropouts = [nn.Dropout(p=config.dropout) for _ in range(len(mlp_layers) - 1)] + [nn.Identity()]
        self.generator = nn.Sequential(*[
            layer for layers in zip(mlp_layers, activations, dropouts) for layer in layers
        ])
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_ID)  # 使用 PAD_ID 作为忽略索引
        self._init_lazy()
        
    def _init_lazy(self):
        dummy_src = torch.zeros((1, 32), dtype=torch.long)
        dummy_tgt = torch.zeros((1, 32), dtype=torch.long)
        self.forward(dummy_src, tgt_ids=dummy_tgt)
    
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
        # tgt_mask = ~(tgt_mask == 0.0).bool()
        
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
            