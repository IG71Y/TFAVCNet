import torch
import torch.nn as nn
from functools import partial
import sub_attention
from vsc_model import vit_base_patch16_224_in21k


class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.linear_q = nn.Linear(input_dim, hidden_dim)
        self.linear_k = nn.Linear(input_dim, hidden_dim)
        self.linear_v = nn.Linear(input_dim, hidden_dim)
        self.linear_o = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        q = self.linear_q(x)
        k = self.linear_k(x)
        v = self.linear_v(x)
        q = q.view(q.size(0), q.size(1), self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(k.size(0), k.size(1), self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(v.size(0), v.size(1), self.num_heads, self.head_dim).transpose(1, 2)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(self.head_dim, dtype=torch.float32))
        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(attn_output.size(0), -1, self.hidden_dim)
        output = self.linear_o(attn_output)
        return output


class MultiHeadAttention2(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention2, self).__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x):
        batch_size = x.size(0)
        k = self.k_linear(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        q = self.q_linear(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.v_linear(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / self.d_k ** 0.5
        scores = torch.softmax(scores, dim=-1)
        output = torch.matmul(scores, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)

        output = self.out(output)

        return output


class MyModel(nn.Module):
    def __init__(self, input_dim, num_classes, num_heads=8, hidden_dim=2048):
        super(MyModel, self).__init__()
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads  # 256
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(p=0.1)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.multihead_attn = nn.MultiheadAttention(hidden_dim, num_heads)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, num_classes)
        self.feed_forward_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x):
        # x shape: (batch_size, input_dim)
        x = self.dropout(x)
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.norm1(x)
        x = x.unsqueeze(1)  # shape: (batch_size,seq_len=1,hidde
        # n_dim)= b,1,2048
        attn_output, _ = self.multihead_attn(x, x, x)  # b,1,2048
        attn_output = attn_output.squeeze(1)  # 4,2048
        x = x.squeeze(1) + attn_output  # 4,2048
        x = self.norm2(x)
        x = self.feed_forward_layer(x)
        # x = self.linear2(x)
        return x


class FusionNet(nn.Module):

    def __init__(self, image_net,audio_net):
        super(FusionNet, self).__init__()

        self.image_net = image_net
        self.audio_net = audio_net
        # self.softmax = nn.Softmax(dim=2)
        # self.encoder = sub_attention.Encoder(1)
        # self.multihead_attn = MyModel(input_dim=2048, num_classes=13)
        # self.self_attn = SelfAttentionClassifier(input_dim=2048, num_classes=13)

        self.classifier_img = nn.Sequential(
            nn.Linear(768, 13)
        )
        self.classifier_aud = nn.Sequential(
            # nn.Dropout(0.2),
            nn.Linear(768, 13)
        )
        self.classifier = nn.Sequential(
            nn.Linear(1536, 13)
        )

    def forward(self, image, audio):

        audio = self.audio_net(audio)
        image = self.image_net(image)
        concat_rep = torch.cat((image, audio), dim=-1)
        # concat_rep = audio + image


        audio_result = self.classifier_aud(audio).unsqueeze(1)
        audio_result = torch.softmax(audio_result, dim=2)

        image_result = self.classifier_img(image).unsqueeze(1)
        image_result = torch.softmax(image_result, dim=2)

        concat_result = self.classifier(concat_rep).unsqueeze(1)
        concat_result = torch.softmax(concat_result, dim=2)
        concat_result = self.classifier(concat_rep)

        # concat_result_all = torch.cat((audio_result, image_result, concat_result), dim=1)  # concat_result_all:[b,3,13]
        # concat_rep = torch.mean(concat_result_all, dim=1) + torch.max(concat_result_all, dim=1)[0]  # [b,13]

        # concat_rep = torch.sum(concat_result_all, dim=1)  # [b,13]
        # concat_rep = torch.softmax(concat_rep,dim=1)

        # pass input through ResNet50 to extract local features
        # audio_rep = self.audio_net(audio)[0]  # audio.size:[b,1,400,64],audio_rep.size:[b,2048]

        # # reshape output to (batch_size, seq_len=1, feature_dim=2048)
        # # reshape output to (batch_size, feature_dim, seq_len)
        # audio_rep = audio_rep.view(audio_rep.size(0), 1, -1)  # audio_rep.size:[b,1,2048]
        # pass output through self-attention mechanism for global processing

        # 2. audio_attention
        # audio_rep = self.multihead_attn(audio_rep)  # b,2048
        # audio_result = self.classifier_aud(audio_rep).unsqueeze(1)  # audio_result : [b,1,13]

        # 3. audio_result
        # audio_result = self.classifier_aud(audio_rep).unsqueeze(1)  # audio_result : [b,1,13]
        # audio_result = self.softmax(audio_result)  # audio_result : [b,1,13]


        # image_rep = self.image_net(image)[0]  # image.size: [b,3,256,256],image_rep.size:[b,2048]

        # image_rep = self.self_attn(image_rep)   # b,2048
        # image_result = self.classifier_img(image_rep).unsqueeze(1)  # image_result: [b,1,13]
        # image_result = self.softmax(image_result)  # b,1,13

        # image_rep = self.image_fc1(image_rep)
        # image_rep = self.relu(image_rep) # batch_size * 1024

        # audio_rep = self.audio_fc1(audio_rep)
        # audio_rep = self.relu(audio_rep) # batch_Size * 1024

        # concat_rep = torch.cat((image_rep, audio_rep), dim=-1)  # concat_rep：[b,4096]
        # concat_rep = image_rep + audio_rep
        # att= concat_rep + audio_rep + image_rep


        # concat_result = self.classifier(concat_rep).unsqueeze(1)  # concat_result:[b,1,13]
        # concat_result = self.classifier(concat_rep)  # concat_result:[b,13]
        # concat_result = self.softmax(concat_result)


        # concat_result_all = torch.cat((audio_result, image_result, concat_result), dim=1)  # concat_result_all:[b,3,13]
        # concat_rep = torch.mean(concat_result_all, dim=1) + torch.max(concat_result_all, dim=1)[0]  # [b,13]

        return concat_result
