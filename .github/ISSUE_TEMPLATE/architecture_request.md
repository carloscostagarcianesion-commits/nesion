---
name: Architecture support request
about: Request support for a new model architecture
labels: architecture, help wanted
---

**Model name:**

**Hugging Face model ID:**

**Attention class name:**
```python
# Run this to find it:
print(model.model.layers[0].self_attn.__class__.__name__)
```

**Are you willing to help test it?** Yes / No
