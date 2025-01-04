from deepface import DeepFace

img1 = "assets/あいう/a.png"
img2 = "assets/あいう/b.png"

result = DeepFace.verify(
    img1_path=img1,
    img2_path=img2,
    model_name="VGG-Face",
    distance_metric="cosine"
)
print("Is verified: ", result["verified"])
