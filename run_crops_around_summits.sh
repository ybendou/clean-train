
python generate_crops_around_summits.py --dataset-path '/home/y17bendo/Documents/datasets/' --dataset miniimagenet --model resnet12 --batch-size 128 --epochs 0 --load-model '/ssd2/backbones/resnet12/miniimagenet/mini1.pt1' --bounding-box-file '/ssd2/data/AugmentedSamples/features/miniImagenet/boundingboxSimplex/AS1000_0123_noPrep_Simplex0.05/boundingbox_summits.pickle' --n-augmentation 50 --save-augmented-features '/ssd2/data/AugmentedSamples/features/miniImagenet/boundingboxSimplex/AS1000_0123_noPrep_Simplex0.05/features0Innerbb.pt' --wandb "brain-imt";

