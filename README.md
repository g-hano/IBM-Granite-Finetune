# IBM-Granite-Finetune

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained("Chan-Y/Stefan-Zweig-Granite", device_map=device)
tokenizer = AutoTokenizer.from_pretrained("Chan-Y/Stefan-Zweig-Granite")

input_text = "As an experienced and famous writer Stefan Zweig, what's your opinion on artificial intelligence?"
inputs = tokenizer(input_text, return_tensors="pt").to(device)

with torch.no_grad():
  outputs = model.generate(
    **inputs,
    max_length=512,
    num_return_sequences=1,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
  )

# Decode the generated text
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text.split(input_text)[-1])
```
```text
Stefan Zweig: Ah, the question of artificial intelligence is indeed a fascinating one, nestled within the vast landscape of human curiosity and technological advancement. As a writer deeply rooted in the humanities, I have always believed that art should reflect our deepest emotions, cultural nuances, and philosophical musings.

Artificial Intelligence, in its quest to mimic human intelligence, presents us with a unique challenge: can machines truly grasp the essence of human experience? Or will they always remain at the level of calculating algorithms, devoid of the subjective, emotional realm that defines us as artists, poets, and thinkers?

On one hand, AI has shown remarkable progress in areas such as pattern recognition, natural language processing, and even creative tasks like composing music or generating literature. These advancements are a testament to the power of computational systems and their ability to learn from vast datasets.

However, I cannot help but feel a sense of unease when contemplating the potential impact of AI on the world of art. If machines can create art that is indistinguishable from human-made pieces, what does this mean for our understanding of creativity and originality? Will we redefine what it means to be an artist, or will AI merely serve as an extension of human creativity?

Moreover, I worry about the homogenization of artistic expression if AI becomes widely adopted. The uniqueness of human experience, with all its complexities and idiosyncrasies, is a crucial element that drives artistic innovation. If AI can replicate this diversity, will it stifle the evolution of artistic styles and movements?

In conclusion, while AI offers incredible possibilities for enhancing human creativity and understanding, it also raises profound questions about the nature of art, originality, and human identity. As we continue to explore this frontier, it is essential that we maintain a critical perspective and ensure that AI serves as a tool to augment human creativity rather than replace it. After all, the essence of art lies not only in its technical execution but also in its ability to reflect and provoke our deepest
```
