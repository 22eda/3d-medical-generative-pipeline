import napari
import nibabel as nib

# İndirdiğin NIfTI dosyasını yükle
pred = nib.load('predicted_ctn.nii.gz').get_fdata()

# Napari görüntüleyicisini başlat
viewer = napari.Viewer()
viewer.add_image(pred, name='Predicted CTn (Model Output)', colormap='gray')

napari.run()