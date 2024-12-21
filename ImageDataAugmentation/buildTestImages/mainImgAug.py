from imgUtils import *
from augmentImages import *
from Helpers import *
from util import *

def augmentImages(sourceFolder, destFolder, rootDir):
    if not os.path.exists(rootDir + "/augworking"):
        os.mkdir(rootDir + "/augworking")
    if not os.path.exists(rootDir + "/augworking/cropedobjects"):
        os.mkdir(rootDir + "/augworking/cropedobjects")

    cropObjectsFromImagePath(sourceFolder + "/images/train",
                             sourceFolder + "/labels/train",
                             rootDir + "/augworking/cropedobjects")

    if not os.path.exists(rootDir + "/augworking/smallObjects"):
        os.mkdir(rootDir + "/augworking/smallObjects")

    generateAugmentedObjects(rootDir + "/augworking/cropedobjects",
                             rootDir + "/augworking/smallObjects")
    if not os.path.exists(destFolder):
        os.mkdir(destFolder)

    copySmallObjectsToOneBlankImage(rootDir + "background.png",
                                    rootDir + "/augworking/smallObjects",
                                    destFolder,
                                    4, 1000)

if __name__ == "__main__":
    print("This is the start of main program of main image augmentation")

    #addLetterBoxesForFolder("/Users/i052090/Downloads/segmentation/data/TSRDMini/train/images",
    #                        "/Users/i052090/Downloads/segmentation/data/TSRDMini/train/labels",
    #                        "/Users/i052090/Downloads/segmentation/data/TSRDMini/letterboxed/images",
    #                        "/Users/i052090/Downloads/segmentation/data/TSRDMini/letterboxed/labels",
    #                        [512,512],[0,0,0])
    rootDir = "E:/ubuntushare/data/warehousetools/"

    if not os.path.exists(rootDir + "/cropedobjects"):
        os.mkdir(rootDir + "/cropedobjects")

    cropObjectsFromImagePath(rootDir + "yolo/train/images",
                             rootDir + "yolo/train/labels",
                             rootDir + "/cropedobjects")

    #copySmallObjectsToOneBlankImage("/Users/i052090/Downloads/roadproject/marks/yolo/blankRoad.png",
    #                                "/Users/i052090/Downloads/roadproject/marks/yolo/objects",
    #                                "/Users/i052090/Downloads/roadproject/marks/yolo/augmented/images",
    #                                3, 2000)

    if not os.path.exists(rootDir + "/smallObjects"):
        os.mkdir(rootDir + "/smallObjects")

    generateAugmentedObjects(rootDir + "/cropedobjects",
                             rootDir + "/smallObjects")
    if not os.path.exists(rootDir + "/augmented"):
        os.mkdir(rootDir + "/augmented")

    copySmallObjectsToOneBlankImage(rootDir + "background.png",
                                    rootDir + "/smallObjects",
                                    rootDir + "/augmented",
                                   4, 5000)

    #addObjRectToImages("/Users/i052090/Downloads/segmentation/data/TSRD/twenty/train/",
    #                   "/Users/i052090/Downloads/segmentation/data/TSRD/twenty/train/annonated")

    #resizeImagesToFixedSize("/Users/i052090/Downloads/segmentation/data/bagspic/marked/original",
    #                        "/Users/i052090/Downloads/segmentation/data/bagspic/marked/resized")

    #copySmallObjectsToManyImages("F:/objectsDetectData/NoSignImages512", "F:/objectsDetectData/allSmallObjects",
    #                             "F:/objectsDetectData/trafficSignTT60/train/images", 10)
    #renamePNGtoNone("F:/objectsDetectData/NoSignImages512")
    #("E:/roadproject/experiment/data/TrafficSign/train/labels")

    #getRetinacsvFormatImagesList("/Users/i052090/Downloads/segmentation/data/TrafficSign/train/images/",
    #                             "/Users/i052090/Downloads/segmentation/data/TrafficSign/train/labels/",
    #                             "./trafficTrain.csv")
    #classesArray = ['i1', 'i10', 'i11', 'i12', 'i13', 'i14', 'i15', 'i2', 'i3', 'i4', 'i5', 'il100', 'il110', 'il50', 'il60', 'il70', 'il80', 'il90', 'io', 'ip', 'p1', 'p10', 'p11', 'p12', 'p13', 'p14', 'p15', 'p16', 'p17', 'p18', 'p19', 'p2', 'p20', 'p21', 'p22', 'p23', 'p24', 'p25', 'p26', 'p27', 'p28', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8', 'p9', 'pa10', 'pa12', 'pa13', 'pa14', 'pa8', 'pb', 'pc', 'pg', 'ph1.5', 'ph2', 'ph2.1', 'ph2.2', 'ph2.4', 'ph2.5', 'ph2.8', 'ph2.9', 'ph3', 'ph3.2', 'ph3.5', 'ph3.8', 'ph4', 'ph4.2', 'ph4.3', 'ph4.5', 'ph4.8', 'ph5', 'ph5.3', 'ph5.5', 'pl10', 'pl100', 'pl110', 'pl120', 'pl15', 'pl20', 'pl25', 'pl30', 'pl35', 'pl40', 'pl5', 'pl50', 'pl60', 'pl65', 'pl70', 'pl80', 'pl90', 'pm10', 'pm13', 'pm15', 'pm1.5', 'pm2', 'pm20', 'pm25', 'pm30', 'pm35', 'pm40', 'pm46', 'pm5', 'pm50', 'pm55', 'pm8', 'pn', 'pne', 'po', 'pr10', 'pr100', 'pr20', 'pr30', 'pr40', 'pr45', 'pr50', 'pr60', 'pr70', 'pr80', 'ps', 'pw2', 'pw2.5', 'pw3', 'pw3.2', 'pw3.5', 'pw4', 'pw4.2', 'pw4.5', 'w1', 'w10', 'w12', 'w13', 'w16', 'w18', 'w20', 'w21', 'w22', 'w24', 'w28', 'w3', 'w30', 'w31', 'w32', 'w34', 'w35', 'w37', 'w38', 'w41', 'w42', 'w43', 'w44', 'w45', 'w46', 'w47', 'w48', 'w49', 'w5', 'w50', 'w55', 'w56', 'w57', 'w58', 'w59', 'w60', 'w62', 'w63', 'w66', 'w8', 'wo', 'i6', 'i7', 'i8', 'i9', 'ilx', 'p29', 'w29', 'w33', 'w36', 'w39', 'w4', 'w40', 'w51', 'w52', 'w53', 'w54', 'w6', 'w61', 'w64', 'w65', 'w67', 'w7', 'w9', 'pax', 'pd', 'pe', 'phx', 'plx', 'pmx', 'pnl', 'prx', 'pwx', 'w11', 'w14', 'w15', 'w17', 'w19', 'w2', 'w23', 'w25', 'w26', 'w27', 'pl0', 'pl4', 'pl3', 'pm2.5', 'ph4.4', 'pn40', 'ph3.3', 'ph2.6']
    #generateRetinaClassesList(classesArray, "./classes.txt")
    #generateIndexFile(221, "./indexMax.txt")

    folderPath = "/Users/i052090/Downloads/segmentation/data/markedhkbridge/coco/VOCAll/annotations"
    #renameAllInFolder(folderPath, "800")
    #renameAllToNumberInFolder(folderPath, 50000)
    #renameTojpgFiles("")
    #makeFileNameAndNameConsistent(folderPath)

    #renameZeroStartFiles("/Users/i052090/Downloads/segmentation/data/markedhkbridge/all")

    #generateXMLListFile("/Users/i052090/Downloads/segmentation/data/markedhkbridge/coco/VOCAll/Annotations", "./hkbridgexml.txt")

    #generateXMLListTrainValFile("/Users/i052090/Downloads/segmentation/data/ydbridge/all/VOCAll/annotations",
    #                            "./ydbridgetrainxml.txt", "./ydbridgevalxml.txt")

    #generateFileNameOnlyDivideTrainValList("/Users/i052090/Downloads/roadproject/marks/centercrop/Annotations/",
    #                        "/Users/i052090/Downloads/roadproject/marks/centercrop/ImageSets/Main/train.txt",
    #                         "/Users/i052090/Downloads/roadproject/marks/centercrop/ImageSets/Main/val.txt",
    #                        1000)
















