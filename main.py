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
    for shirt in shirts:
        p = pant
        s = shirt
        s = FormatPhoto.makeImage(f"clothes/shirts/{s}")
        p = FormatPhoto.makeImage(f"clothes/pants/{p}")
        s = FormatPhoto.rescaleImage(s)
        p = FormatPhoto.rescaleImage(p)

        # if p not in saved:
        #     cv2.imwrite(f"clothes/pairs/{pant}", p)
        # if s not in saved:
        #     cv2.imwrite(f"clothes/pairs/{pant}", s)

        s = FormatPhoto.noiseReduction(s)
        p = FormatPhoto.noiseReduction(p)
        # if p not in saved:
        #     cv2.imwrite(f"clothes/pairs/{pant}", p)
        # if s not in saved:
        #     cv2.imwrite(f"clothes/pairs/{pant}", s)

        s = FormatPhoto.findAndCut(s, "shirt")
        p = FormatPhoto.findAndCut(p, "")
        # if p not in saved:
        #     cv2.imwrite(f"clothes/pairs/{pant}", p)
        # if s not in saved:
        #     cv2.imwrite(f"clothes/pairs/{pant}", s)
        newPic = FormatPhoto.stitch(s, p)
        name = pant[0:pant.index(".")] + shirt[0:shirt.index(".")]
        cv2.imwrite(f"clothes/pairs/{name}.jpg", newPic)



