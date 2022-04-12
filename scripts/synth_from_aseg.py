import nibabel as nib
import matplotlib.pyplot as plt
import torch
import numpy as np
from b0.synth import labels_to_chi, chi_to_fieldmap, fieldmap_to_shift


root = '/autofs/cluster/vxmdata1/FS_Slim/proc/cleaned'
subj = f'{root}/OASIS_OAS1_0001_MR1'
aseg = f'{subj}/aseg.mgz'


dat = nib.load(aseg)
dat = np.asarray(dat.dataobj).squeeze()
dat = torch.as_tensor(dat)
dat = dat.permute([1, 0, 2])

chi = labels_to_chi(dat > 0)
fmap = chi_to_fieldmap(chi, dim=3, zdim=-1)
dmap = fieldmap_to_shift(fmap)


plt.subplot(2, 2, 1)
plt.imshow(dat[len(dat)//2] > 0)
plt.colorbar()
plt.axis('off')
plt.title('Labels')
plt.subplot(2, 2, 2)
plt.imshow(chi[len(chi)//2])
plt.colorbar()
plt.axis('off')
plt.title('Delta Chi (ppm)')
plt.subplot(2, 2, 3)
plt.imshow(fmap[len(chi)//2])
plt.colorbar()
plt.axis('off')
plt.title('Delta B0 (Hz)')
plt.subplot(2, 2, 4)
plt.imshow(dmap[len(chi)//2])
plt.colorbar()
plt.axis('off')
plt.title('Displacement (pix)')
plt.show()


