from torch import nn
class ViltForVQA(nn.Module):
    def __init__(self, peft_model, num_labels):
        super().__init__()
        self.vilt = peft_model.base_model.model.vilt
        self.num_labels = num_labels
        self.classifier = nn.Sequential(
            nn.Linear(self.vilt.config.hidden_size, self.vilt.config.hidden_size * 2),
            nn.LayerNorm(self.vilt.config.hidden_size * 2, eps=1e-5),
            nn.GELU(),
            nn.Linear(self.vilt.config.hidden_size * 2, num_labels)
        )

    def forward(self, pixel_values, input_ids, attention_mask, token_type_ids=None, labels=None):
        outputs = self.vilt(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        pooled_output = outputs.pooler_output
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)

        return {"logits": logits, "loss": loss}