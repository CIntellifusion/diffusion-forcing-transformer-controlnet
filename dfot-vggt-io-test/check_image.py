import cv2 
p1 = ''
p2 = ''

i1 = cv2.imread(p1)
i2 = cv2.imread(p2)

print(i1.shape, i2.shape)
print((i1 == i2).all())
print((i1 - i2).mean())
print((i1 - i2).max())

# image tensor from pt : torch.Size([1, 3, 518, 518]) 1.0  0.0 0.4204689860343933
# image tensor from image: torch.Size([1, 3, 518, 518]) 1.0  0.0 0.4204752445220947