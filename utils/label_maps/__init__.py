from .lm_ch import get_lr_to_4_cls, get_hr_to_4_cls, get_17_cls_to_4_cls, get_lr_to_17_cls, get_17_cls_to_lr
from .lm_pl import get_lr_and_train_map, get_train_to_4_cls_label_map, get_lr_to_4_cls_label_map

__all__ = ['get_xx_label_map']


def get_xx_label_map(in_type='', out_type='', ):
    '''
        Literal[
        'nlcd_label', 'lc_label',
        'Target_4_cls',
        'nlcd_label_train', 'lc_label_train',
        ]
    '''

    if out_type == 'Target_4_cls':
        if in_type == 'nlcd_label':  # LR label
            return get_lr_to_4_cls()

        if in_type == 'lc_label':  # HR label
            return get_hr_to_4_cls()

        if in_type == 'nlcd_label_train':  # LR Train label
            return get_17_cls_to_4_cls()

    if in_type == 'nlcd_label_train' and out_type == 'nlcd_label':
        return get_17_cls_to_lr()

    if in_type == 'nlcd_label' and out_type == 'nlcd_label_train':
        return get_lr_to_17_cls()

    '''
        Literal[
        'ESA_GLC10_label', 'Esri_GLC10_label', 'FORM_GLC10_label', 'GLC_FCS30_label',
         'HR_ground_truth', 'Target_4_cls',
        'ESA_GLC10_label_train', 'Esri_GLC10_label_train', 'FORM_GLC10_label_train', 'GLC_FCS30_label_train'
        ]
    '''
    if '_train' in out_type and in_type in ['ESA_GLC10_label', 'Esri_GLC10_label', 'FORM_GLC10_label',
                                            'GLC_FCS30_label', 'HR_ground_truth']:
        return get_lr_and_train_map(in_type)[0]

    if '_train' in in_type and out_type in ['ESA_GLC10_label', 'Esri_GLC10_label', 'FORM_GLC10_label',
                                            'GLC_FCS30_label', 'HR_ground_truth']:
        return get_lr_and_train_map(out_type)[-1]

    if out_type == 'Target_4_cls':
        if '_train' in in_type:
            return get_train_to_4_cls_label_map(in_type)
        else:
            return get_lr_to_4_cls_label_map(in_type)

    raise ValueError("function 'get_xx_label_map' get an invalid input parameter", in_type, out_type)
