import nibabel as nib
import matplotlib.pyplot as plt
from b0.synth import labels_to_chi, chi_to_fieldmap, fieldmap_to_shift


root = '/autofs/cluster/vxmdata1/proc/cleaned'
subj = f'{root}/OASIS_OAS1_0001_MRI'
aseg = f'{subj}/aseg.mgz'


dat = nib.load(aseg)
dat = dat.get_data().squeeze()

chi = labels_to_chi(dat > 0)
fmap = chi_to_fieldmap(chi, dim=3, zdim=-1)
dmap = fieldmap_to_shift(fmap)


plt.subplot(2, 2, 1)
plt.imshow(dat[len(dat)//2])
plt.colorbar()
plt.axis('off')
plt.title('Labels')
plt.subplot(2, 2, 2)
plt.imshow(chi[len(chi)//2])
plt.colorbar()
plt.axis('off')
plt.title('Chi')
plt.subplot(2, 2, 3)
plt.imshow(fmap[len(chi)//2])
plt.colorbar()
plt.axis('off')
plt.title('B0')
plt.subplot(2, 2, 4)
plt.imshow(dmap[len(chi)//2])
plt.colorbar()
plt.axis('off')
plt.title('Displacement')


