from roboflow import Roboflow
rf = Roboflow(api_key="4OleiZlXQuI6r36DDEEL")
project = rf.workspace("thnhan1149").project("hand-classification-aiokx")
version = project.version(5)
dataset = version.download("folder")
                