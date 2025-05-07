import torch
from torch import nn


class ClassEmbedAdapter(nn.Module):
    def __init__(self, args, class_labels):
        super().__init__()
        self.args = args
        self.class_labels = class_labels
        self.prompt_embeds = self.encode_class_names(self.class_labels)
        self.text_embedding_dim = self.prompt_embeds.shape[-1]
        self.projection_class_embeddings_input_dim = 1280
        self.single_class_embedding_dim = 256
        self.class_embedding_count = self.projection_class_embeddings_input_dim // self.single_class_embedding_dim

        self.adapter = nn.Sequential(
            nn.Linear(self.text_embedding_dim, self.single_class_embedding_dim),
            nn.SiLU(),
            nn.Linear(self.single_class_embedding_dim, self.single_class_embedding_dim),
        )

    @torch.no_grad()
    def encode_class_names(self, class_labels):
        from transformers import AutoTokenizer, CLIPTextModelWithProjection
        class_names = class_labels.names
        tokenizer = AutoTokenizer.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", subfolder="tokenizer", local_files_only=True)
        tokenizer_2 = AutoTokenizer.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", subfolder="tokenizer_2", local_files_only=True)
        text_encoder = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-large-patch14", local_files_only=True)
        text_encoder_2 = CLIPTextModelWithProjection.from_pretrained("laion/CLIP-ViT-bigG-14-laion2B-39B-b160k", local_files_only=True)
        text_encoder.to('cuda', dtype=torch.float16)
        text_encoder_2.to('cuda', dtype=torch.float16)

        prompt_embeds_list = []
        tokenizers = [tokenizer, tokenizer_2]
        text_encoders = [text_encoder, text_encoder_2]
        for tokenizer, text_encoder in zip(tokenizers, text_encoders):
            text_input_ids = tokenizer(class_names, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt").input_ids
            pooled_prompt_embeds = text_encoder(text_input_ids.to(text_encoder.device), return_dict=False)[0]
            prompt_embeds_list.append(pooled_prompt_embeds)
        prompt_embeds = torch.cat(prompt_embeds_list, dim=-1)
        return prompt_embeds

    def forward(self, class_name):
        assert isinstance(class_name, list)
        class_name = [["None"] if not name else name for name in class_name]
        label_ids = [self.class_labels.str2int(name) for name in class_name]
        class_emb = [self.prompt_embeds[label_id] for label_id in label_ids]
        # pad class embeddings to the same size
        for i in range(len(class_emb)):
            if class_emb[i].shape[0] <= self.class_embedding_count:
                class_emb[i] = torch.cat([class_emb[i], self.prompt_embeds[0].unsqueeze(0).repeat(self.class_embedding_count - class_emb[i].shape[0], 1)], dim=0)
            else:
                class_emb[i] = class_emb[i][torch.randperm(class_emb[i].shape[0])[:self.class_embedding_count]]
        class_emb = torch.stack(class_emb)

        # format class embeddings
        class_emb = self.adapter(class_emb)
        bsz, _, _ = class_emb.shape
        class_emb = class_emb.view(bsz, -1)
        return class_emb
