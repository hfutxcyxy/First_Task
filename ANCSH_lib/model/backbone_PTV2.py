from torch import nn
from point_transformer_v2m2_base import Encoder,Decoder,BlockSequence,UnpoolWithSkip
from point_transformer_v2m2_base import GVAPatchEmbed
from point_transformer_v2m2_base import PointBatchNorm
class PTV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = GVAPatchEmbed(
            in_channels=12,
            embed_channels=48,
            groups=6,
            depth=1,
            neighbours=8,
            qkv_bias=True,
            pe_multiplier=False,
            pe_bias=True,
            attn_drop_rate=0.0,
            enable_checkpoint=False,
        )

        # 简化PTV2网络，两个encoder、一个decoder，中间一个MLP，最后一个卷积层
        self.encoder1 = Encoder(
            in_channels=48,
            grid_size=0.06,
            depth=2,
            embed_channels=96,
            groups=12,
            neighbours=16,
            qkv_bias=True,
            pe_multiplier=False,
            pe_bias=True,
            attn_drop_rate=0.0,
            drop_path_rate=0.0,
            enable_checkpoint=False,
        )

        self.encoder2 = Encoder(
            in_channels=96,
            grid_size=0.12,
            depth=2,
            embed_channels=192,
            groups=24,
            neighbours=16,
            qkv_bias=True,
            pe_multiplier=False,
            pe_bias=True,
            attn_drop_rate=0.0,
            drop_path_rate=0.0,
            enable_checkpoint=False,
        )

        self.decoder = Decoder(
            depth=1,
            in_channels=192,
            skip_channels=96,
            embed_channels=128,
            groups=48,
            neighbours=16,
            qkv_bias=True,
            pe_multiplier=False,
            pe_bias=True,
            attn_drop_rate=0.0,
            drop_path_rate=0,
            enable_checkpoint=False,
            unpool_backend="map",
        )

        self.mid_connection = nn.Sequential(
            nn.Linear(192,192),
            PointBatchNorm(192),
            nn.ReLU(192),
            nn.Linear(192,192)
        )

        self.fc_layer = nn.Sequential(
            nn.Conv1d(48, 128, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(True),
            nn.Dropout(0.5)
        )

    def forward(self, input):
        l0_coord = input[:, :, :3]
        l0_feat = input[:, :, 3:]
        l0_points = [l0_coord,l0_feat]
        l0_points = self.patch_embed(l0_points)
        l1_points,cluster = self.encoder1(l0_points)
        skip_cluster = cluster
        l2_points,cluster = self.encoder2(l1_points)
        l3_points = self.mid_connection(l2_points)
        #encoder_1与decoder之间有残差连接
        l4_points = self.decoder(l3_points,l1_points,skip_cluster)
        coord,feat = l4_points
        return self.fc_layer(feat)


