# Description:
1. Get text from images. Get both the text that is annotating the image (usually top and bottom) and text hidden in the image (like the background or if the image is a meme)
2. Check if just the text is misogynistic.
3. Look at just the image, ignoring the text. Transcribe the image into text.
4. Check if the descriptive text is misogynistic.
5. As an additional step, point out, that there might be a woman or something descibing a woman in the image.
6. Combine the text description of the image and the text from step 1 and see if that combined text is misogynistic.
7. For all images seen as misogynistic, do Task B of desciding which of the multiple classes of misogyny are present.

# Further work
1. Use LLM as a judge
2. Add adversarial text samples
