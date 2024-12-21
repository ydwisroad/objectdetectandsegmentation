import torchvision.transforms
import PIL.Image as Image
import torchvision.transforms
import os
import tqdm


def centerCropImages(src_images_dir, dest_images_dir, size=(512,512)):
    images = os.listdir(src_images_dir)

    for img_name in images:
        image =Image.open(src_images_dir + "/" + img_name)
        print(image.size, image.format, image.mode)

        width = image.size[0]
        height = image.size[1]

        print(width)
        print(height)
        # 生成一个CenterCrop类的对象,用来将图片从中心裁剪成 size
        crop_obj = torchvision.transforms.CenterCrop((size, size))
        image = crop_obj(image)

        resize_obj = torchvision.transforms.Resize(size)
        image = resize_obj(image)

        # 将裁剪之后的图片保存下来
        image.save(dest_images_dir + "/" + img_name, format='PNG')

def centerCropImage(inputFolder, outputFolder, size = (512,512)):

    images = os.listdir(inputFolder)

    for img_name in images:
        # 读入图片
        image = Image.open(inputFolder + "/" + img_name)
        print(image.size, image.format, image.mode)

        # 生成一个CenterCrop类的对象,用来将图片从中心裁剪成224*224
        crop_obj = torchvision.transforms.CenterCrop(size)
        image = crop_obj(image)

        # 将裁剪之后的图片保存下来
        image.save(outputFolder + "/" + img_name, format='PNG')

if __name__ == "__main__":
    src_images_dir = "E:/newpaper2024/data/badweather/original/"
    dest_images_dir = "E:/newpaper2024/data/badweather/centerresize/"
    if not os.path.exists(dest_images_dir):
        os.mkdir(dest_images_dir)
    #centerCropImages(src_images_dir, dest_images_dir)
    centerCropImage(src_images_dir, dest_images_dir)

