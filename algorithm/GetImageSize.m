function sz=GetImageSize(impath)
info = imfinfo(impath);
sz = [info(1).Height info(1).Width];
