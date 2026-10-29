from nilearn import image
import os
import itertools

stim_sites = ['DLPFC', 'PSTS', 'RIFG']
stim_types = ['Active', 'Sham']
parcellations = ['Schaefer100+Tian+Suit', 'Schaefer200+Tian+Suit']
sessions = [1,2]

for parcellation in parcellations:
    source = image.load_img('data/source_atlases/' + parcellation + '.nii.gz')

    for site, stim_type, ses in itertools.product(stim_sites, stim_types, sessions):
        for sub in os.scandir('data/' + site + '/' + stim_type + '/'):

            atlas_dir = sub.path + '/ses-0' + str(ses) + '/atlases'
            if not os.path.isdir(atlas_dir):
                os.mkdir(atlas_dir)

            if site == 'DLPFC' or stim_type == 'RIFG':
                ref_name = 'sub-' + sub.name[1:] + '_ses-0' + str(ses) + '_T1w.nii'
            elif site == 'PSTS':
                ref_name = sub.name[1:] + '_0' + str(ses) + '_T1_ns_atnl.nii.gz'

            ref_path = sub.path + '/ses-0' + str(ses) + '/sMRI_T1/' + ref_name
            new_atlas_path = atlas_dir + '/' + sub.name + '_' + parcellation + '.nii.gz'

            if os.path.isfile(ref_path) and not os.path.isfile(new_atlas_path):
                print(new_atlas_path)
                ref = image.load_img(ref_path)
                new_atlas = image.resample_to_img(source, ref, interpolation = 'nearest')
                new_atlas.to_filename()


        