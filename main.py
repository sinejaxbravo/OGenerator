import os
import cv2
import FormatPhoto

shirts = []
pants = []
saved = ['IMG_0663.jpg', 'IMG_0655.jpg', 'IMG_0664.jpg']
for filename in os.listdir("clothes/shirts"):
    shirts.append(filename)

for filename in os.listdir("clothes/pants"):
    pants.append(filename)


print(pants, "    shirts ", shirts)

for pant in pants:
    p = pant
    p = FormatPhoto.makeImage(f"clothes/pants/{p}")
    p = FormatPhoto.rescaleImage(p)
    p = FormatPhoto.noiseReduction(p)
    p = FormatPhoto.findAndCut(p, "")
    for shirt in shirts:

        s = shirt
        s = FormatPhoto.makeImage(f"clothes/shirts/{s}")
        s = FormatPhoto.rescaleImage(s)
        s = FormatPhoto.noiseReduction(s)
        s = FormatPhoto.findAndCut(s, "shirt")

        newPic = FormatPhoto.stitch(s, p)
        name = pant[0:pant.index(".")] + shirt[0:shirt.index(".")]
        cv2.imwrite(f"clothes/pairs/{name}.jpg", newPic)



