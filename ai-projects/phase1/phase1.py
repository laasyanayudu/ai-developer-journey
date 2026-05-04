model_name= "GPT-4"
my_name= "Alex"
accuracy= 95.7
num_layers= 12
is_trained= True
scores=[88, 92, 79, 95, 61]
model_info={"name":"BERT",
            "accuracy":94.2,
            "trained": True}
print("Model:", model_name)
print("Accuracy:",accuracy)
print("Scores:", scores)
print("First Score:", scores[0])
print("Model name:", model_info["name"])
languages=["Python", "Javascript", "Rus"]
print("I am learning", languages[0])
print("Total languages", len(languages))
def greet_model(name, accuracy):
    message= "Model: " + name + " | Accuracy: " + str(accuracy)+ "%"
    return message
print(greet_model("BERT", 94.2))
print(greet_model("GPT-4", 97.5))
print(greet_model("ResNet", 89.1))
for score in scores:
    print("Score:", score)
def check_score(score):
    if score>=90:
     return "Excellent" 
    elif score>=75:
       return "Good"
    else:
       return "Needs improvement"
for score in scores:
    result=check_score(score)
    print(score,  "->", result)