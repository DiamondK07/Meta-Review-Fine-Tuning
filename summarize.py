from transformers import T5ForConditionalGeneration, T5TokenizerFast, pipeline

# Example Meta Review Text
meta_review_text = """
This paper presents a novel approach to utilizing large-scale human video data for training robot manipulation skills. The authors propose a method that extracts relevant motion data from publicly available internet videos to improve robotic dexterity and control. The methodology includes a pre-processing pipeline for extracting human hand motion trajectories, followed by training a deep learning model that translates these movements into robot-compatible instructions. The authors conduct experiments using multiple real-world tasks, such as object picking, pushing, and rotating, showing that their approach can achieve high accuracy in mimicking human hand movements. The strengths of the paper include its innovative use of publicly available video data, the integration of multiple components to form a cohesive system, and the validation on real-world tasks. 

However, several weaknesses are noted by the reviewers. One primary concern is the lack of sufficient quantitative results comparing the new method to existing approaches. While the authors provide qualitative demonstrations, the reviewers argue that these are not enough to validate the effectiveness of the method comprehensively. Additionally, the method's reliance on specific types of video data that may not be universally available is highlighted as a limitation. The reviewers also point out that while the method is novel, it is not entirely clear how it performs on more complex manipulation tasks that require precise and fine-grained control.

The reviewers commend the authors for their thorough literature review, as they accurately position their work within the context of existing research and identify gaps that their approach attempts to fill. Furthermore, the paper's discussion on future work, including the extension of the model to other domains such as autonomous vehicle navigation, is seen as a promising direction. Despite these positive aspects, the reviewers recommend that the authors include additional experiments that provide a more comprehensive performance analysis. They also suggest that a section be added discussing the limitations of the approach in detail, specifically addressing the types of tasks for which the method may not be suitable. 

In conclusion, while the paper demonstrates a clear contribution to the field of robot learning and manipulation, there are areas where improvements are needed. If the authors address the concerns raised—especially by providing quantitative evidence of their model’s effectiveness and exploring its applicability to a wider range of manipulation tasks—the reviewers believe that the paper could be a valuable addition to the literature. Therefore, they assign a borderline accept score with suggestions for improvement in a subsequent revision.
"""

# Initialize the tokenizer and model from the saved directory
tokenizer = T5TokenizerFast.from_pretrained('./results/final_model')
model = T5ForConditionalGeneration.from_pretrained('./results/final_model')

# Initialize the summarization pipeline with the fine-tuned model
summarizer = pipeline("summarization", model=model, tokenizer=tokenizer, device=0)

# Prepare the prompt for the summarization
prompt = (
    "Summarize the following paper meta review in a concise and informative manner.\n"
    "Focus on the key points, strengths, and weaknesses mentioned in the review, and provide an overview that captures "
    "the essence of the analysis. Ensure the summary highlights the core contributions and any major critiques.\n\n"
    f"Meta Review: {meta_review_text}\nSummary:"
)

# Generate the summary
summary = summarizer(prompt, max_length=150, min_length=50, truncation=True)

# Print the generated summary
print("Generated Summary:", summary[0]['summary_text'])
