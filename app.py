from flask import Flask, render_template, request
import torch
import torch.nn as nn

app = Flask(__name__)
with open("pytorch_data/best_params.txt", "r") as f:
    n1, n2, n3 = map(int, f.read().split(","))

model = nn.Sequential(
    nn.Linear(1, n1), nn.ReLU(),
    nn.Linear(n1, n2), nn.ReLU(),
    nn.Linear(n2, n3), nn.ReLU(),
    nn.Linear(n3, 2)
)
model.load_state_dict(torch.load("pytorch_data/pytorch.pth"))
model.eval()

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        hours = float(request.form["hours"])
        input_tensor = torch.tensor([[hours]], dtype=torch.float32)
        output = model(input_tensor)
        marks = round(output[0][0].item(), 2)
        grade = round(output[0][1].item(), 2)
        result = f"📘 Marks: {marks}, 🎓 Grade: {grade}"
    return render_template("front_code.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
