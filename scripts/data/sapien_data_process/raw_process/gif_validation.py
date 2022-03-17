from tkinter import *
from PIL import Image, ImageTk
import json
import glob
from functools import partial


DATA_DIR = "/home/hja40/Desktop/Dataset/saipen_articulations/"
WIDTH = 512
HEIGHT = 512

CONTINUE_PART_NUM = 80


class MyLabel(Label):
    def __init__(self, master, filename):
        im = Image.open(filename)
        seq = []
        try:
            while 1:
                seq.append(im.copy())
                im.seek(len(seq))  # skip to next frame
        except EOFError:
            pass  # we're done

        try:
            self.delay = im.info["duration"]
        except KeyError:
            self.delay = 100

        first = seq[0].convert("RGBA").resize([WIDTH, HEIGHT])
        self.frames = [ImageTk.PhotoImage(first)]

        Label.__init__(self, master, image=self.frames[0])

        temp = seq[0]
        for image in seq[1:]:
            temp.paste(image)
            frame = temp.convert("RGBA").resize([WIDTH, HEIGHT])
            self.frames.append(ImageTk.PhotoImage(frame))

        self.idx = 0

        self.cancel = self.after(self.delay, self.play)

    def play(self):
        self.config(image=self.frames[self.idx])
        self.idx += 1
        if self.idx == len(self.frames):
            self.idx = 0
        self.cancel = self.after(self.delay, self.play)


def isDrawer(window, modelCat, modelId, partIndex):
    window.destroy()
    new_data = f"{modelCat} {modelId} {partIndex} drawer\n"
    print("This part has been annotated as a drawer")

    # Promise that data is written into files after each epoch
    validPartFile = open("validPart.txt", "a+")
    validPartFile.write(new_data)
    validPartFile.close()
    print("\n")

def isDoor(window, modelCat, modelId, partIndex):
    window.destroy()
    new_data = f"{modelCat} {modelId} {partIndex} door\n"
    print("This part has been annotated as a door")

    # Promise that data is written into files after each epoch
    validPartFile = open("validPart.txt", "a+")
    validPartFile.write(new_data)
    validPartFile.close()
    print("\n")

def isLid(window, modelCat, modelId, partIndex):
    window.destroy()
    new_data = f"{modelCat} {modelId} {partIndex} lid\n"
    print("This part has been annotated as a lid")

    # Promise that data is written into files after each epoch
    validPartFile = open("test.txt", "a+")
    validPartFile.write(new_data)
    validPartFile.close()
    print("\n")
    

if __name__ == "__main__":
    partFile = open("careModelPart.json")
    carePart = json.load(partFile)
    partFile.close()

    curModelNum = 0
    curPartNum = 0
    for modelCat in carePart.keys():
        for modelId in carePart[modelCat].keys():
            curModelNum += 1
            for partIndex in carePart[modelCat][modelId]:
                curPartNum += 1
                if(curPartNum < CONTINUE_PART_NUM):
                    continue
                print(f"\nThis is the {curModelNum} model and {curPartNum} part:")
                print(f"{modelCat}  {modelId} {partIndex}")
                gif = glob.glob(f"{DATA_DIR}{modelId}/{modelId}-{partIndex}-*.gif")
                # There should be only one gif for each articulated part
                if len(gif) > 1:
                    print("More than one articualtion gif for this part!")
                    exit()
                # If there is no gif for this part, it means that this part is not an articulated part
                if len(gif) == 0:
                    print("This part cannot move, it has been deleted")
                    print("\n")
                    continue

                # If there is only one gif, show the gif and verify this part based on the user input
                gif_file = gif[0]

                window = Tk()
                window.title(gif_file)
                window.geometry(f"{WIDTH}x{HEIGHT+120}")
                anim = MyLabel(window, gif_file)
                anim.pack()

                Button(window, text="Invalid", command=window.destroy).pack()
                Button(window, text="Drawer", command=partial(isDrawer, window, modelCat, modelId, partIndex)).pack()
                Button(window, text="Door", command=partial(isDoor, window, modelCat, modelId, partIndex)).pack()
                Button(window, text="Lid", command=partial(isLid, window, modelCat, modelId, partIndex)).pack()
                window.mainloop()

    print("Finish all the verfication")
