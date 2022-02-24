
#mkdir /ssd2/data/AugmentedSamples/images/miniImagenet/AS300/miniAS50backbone11;
#mkdir /ssd2/data/AugmentedSamples/images/miniImagenet/AS300/miniAS100backbone11;
#mkdir /ssd2/data/AugmentedSamples/images/miniImagenet/AS300/miniAS150backbone11;
#mkdir /ssd2/data/AugmentedSamples/images/miniImagenet/AS300/miniAS200backbone11;
#mkdir /ssd2/data/AugmentedSamples/images/miniImagenet/AS300/miniAS250backbone11;

#mkdir /ssd2/data/AugmentedSamples/images/miniImagenet/AS300/miniAS50backbone21;
#mkdir /ssd2/data/AugmentedSamples/images/miniImagenet/AS300/miniAS100backbone21;
#mkdir /ssd2/data/AugmentedSamples/images/miniImagenet/AS300/miniAS150backbone21;
#mkdir /ssd2/data/AugmentedSamples/images/miniImagenet/AS300/miniAS200backbone21;
#mkdir /ssd2/data/AugmentedSamples/images/miniImagenet/AS300/miniAS250backbone21;

#python main.py --dataset-path '../../../datasets/' --dataset miniimagenet --model resnet12 --batch-size 128 --epochs 0 --load-model '/ssd2/backbones/resnet12/miniimagenet/mini1.pt1' --n-augmentation 50 --save-augmented-features '/ssd2/data/AugmentedSamples/features/miniImagenet/AS300/miniAS50backbone11.pt' --save-images '/ssd2/data/AugmentedSamples/images/miniImagenet/AS300/miniAS50backbone11' --seed 666666 ;
#python main.py --dataset-path '../../../datasets/' --dataset miniimagenet --model resnet12 --batch-size 128 --epochs 0 --load-model '/ssd2/backbones/resnet12/miniimagenet/mini2.pt1' --n-augmentation 50 --save-augmented-features '/ssd2/data/AugmentedSamples/features/miniImagenet/AS300/miniAS50backbone21.pt' --save-images '/ssd2/data/AugmentedSamples/images/miniImagenet/AS300/miniAS50backbone21' --seed 666666 ;
#python main.py --dataset-path '../../../datasets/' --dataset miniimagenet --model resnet12 --batch-size 128 --epochs 0 --load-model '/ssd2/backbones/resnet12/miniimagenet/mini3.pt1' --n-augmentation 50 --save-augmented-features '/ssd2/data/AugmentedSamples/features/miniImagenet/AS300/miniAS50backbone31.pt' --seed 666666;


#python main.py --dataset-path '../../../datasets/' --dataset miniimagenet --model resnet12 --batch-size 128 --epochs 0 --load-model '/ssd2/backbones/resnet12/miniimagenet/mini1.pt1' --n-augmentation 50 --save-augmented-features '/ssd2/data/AugmentedSamples/features/miniImagenet/AS300/miniAS100backbone11.pt' --save-images '/ssd2/data/AugmentedSamples/images/miniImagenet/AS300/miniAS100backbone11' --seed 555555 ;
#python main.py --dataset-path '../../../datasets/' --dataset miniimagenet --model resnet12 --batch-size 128 --epochs 0 --load-model '/ssd2/backbones/resnet12/miniimagenet/mini2.pt1' --n-augmentation 50 --save-augmented-features '/ssd2/data/AugmentedSamples/features/miniImagenet/AS300/miniAS100backbone21.pt' --save-images '/ssd2/data/AugmentedSamples/images/miniImagenet/AS300/miniAS100backbone21' --seed 555555 ;
#python main.py --dataset-path '../../../datasets/' --dataset miniimagenet --model resnet12 --batch-size 128 --epochs 0 --load-model '/ssd2/backbones/resnet12/miniimagenet/mini3.pt1' --n-augmentation 50 --save-augmented-features '/ssd2/data/AugmentedSamples/features/miniImagenet/AS300/miniAS100backbone31.pt' --seed 555555;


#python main.py --dataset-path '../../../datasets/' --dataset miniimagenet --model resnet12 --batch-size 128 --epochs 0 --load-model '/ssd2/backbones/resnet12/miniimagenet/mini1.pt1' --n-augmentation 50 --save-augmented-features '/ssd2/data/AugmentedSamples/features/miniImagenet/AS300/miniAS150backbone11.pt' --save-images '/ssd2/data/AugmentedSamples/images/miniImagenet/AS300/miniAS150backbone11' --seed 888888 ;
#python main.py --dataset-path '../../../datasets/' --dataset miniimagenet --model resnet12 --batch-size 128 --epochs 0 --load-model '/ssd2/backbones/resnet12/miniimagenet/mini2.pt1' --n-augmentation 50 --save-augmented-features '/ssd2/data/AugmentedSamples/features/miniImagenet/AS300/miniAS150backbone21.pt' --save-images '/ssd2/data/AugmentedSamples/images/miniImagenet/AS300/miniAS150backbone21' --seed 888888 ;
#python main.py --dataset-path '../../../datasets/' --dataset miniimagenet --model resnet12 --batch-size 128 --epochs 0 --load-model '/ssd2/backbones/resnet12/miniimagenet/mini3.pt1' --n-augmentation 50 --save-augmented-features '/ssd2/data/AugmentedSamples/features/miniImagenet/AS300/miniAS150backbone31.pt' --seed 888888;


#python main.py --dataset-path '../../../datasets/' --dataset miniimagenet --model resnet12 --batch-size 128 --epochs 0 --load-model '/ssd2/backbones/resnet12/miniimagenet/mini1.pt1' --n-augmentation 50 --save-augmented-features '/ssd2/data/AugmentedSamples/features/miniImagenet/AS300/miniAS200backbone11.pt' --save-images '/ssd2/data/AugmentedSamples/images/miniImagenet/AS300/miniAS200backbone11' --seed 777777 ;
#python main.py --dataset-path '../../../datasets/' --dataset miniimagenet --model resnet12 --batch-size 128 --epochs 0 --load-model '/ssd2/backbones/resnet12/miniimagenet/mini2.pt1' --n-augmentation 50 --save-augmented-features '/ssd2/data/AugmentedSamples/features/miniImagenet/AS300/miniAS200backbone21.pt' --seed 777777 ;
python main.py --dataset-path '../../../datasets/' --dataset miniimagenet --model resnet12 --batch-size 128 --epochs 0 --load-model '/ssd2/backbones/resnet12/miniimagenet/mini3.pt1' --n-augmentation 50 --save-augmented-features '/ssd2/data/AugmentedSamples/features/miniImagenet/AS300/miniAS200backbone31.pt' --seed 777777;
#python main.py --dataset-path '../../../datasets/' --dataset miniimagenet --model resnet12 --batch-size 128 --epochs 0 --load-model '/ssd2/backbones/resnet12/miniimagenet/mini3.pt1' --n-augmentation 50 --save-augmented-features '/ssd2/data/AugmentedSamples/features/miniImagenet/AS300/miniAS250backbone31.pt' --seed 444444;
#python main.py --dataset-path '../../../datasets/' --dataset miniimagenet --model resnet12 --batch-size 128 --epochs 0 --load-model '/ssd2/backbones/resnet12/miniimagenet/mini3.pt1' --n-augmentation 50 --save-augmented-features '/ssd2/data/AugmentedSamples/features/miniImagenet/AS300/miniAS300backbone31.pt' --seed 333333




### Run these some day
#python main.py --dataset-path '../../../datasets/' --dataset miniimagenet --model resnet12 --batch-size 128 --epochs 0 --load-model '/ssd2/backbones/resnet12/miniimagenet/mini1.pt1' --n-augmentation 50 --save-augmented-features '/ssd2/data/AugmentedSamples/features/miniImagenet/AS300/miniAS250backbone11.pt' --save-images '/ssd2/data/AugmentedSamples/images/miniImagenet/AS300/miniAS250backbone11' --seed 444444 ;
#python main.py --dataset-path '../../../datasets/' --dataset miniimagenet --model resnet12 --batch-size 128 --epochs 0 --load-model '/ssd2/backbones/resnet12/miniimagenet/mini2.pt1' --n-augmentation 50 --save-augmented-features '/ssd2/data/AugmentedSamples/features/miniImagenet/AS300/miniAS250backbone21.pt'--seed 444444 ;
#python main.py --dataset-path '../../../datasets/' --dataset miniimagenet --model resnet12 --batch-size 128 --epochs 0 --load-model '/ssd2/backbones/resnet12/miniimagenet/mini1.pt1' --n-augmentation 50 --save-augmented-features '/ssd2/data/AugmentedSamples/features/miniImagenet/AS300/miniAS300backbone11.pt' --save-images '/ssd2/data/AugmentedSamples/images/miniImagenet/AS300/miniAS300backbone11' --seed 333333 ;
#python main.py --dataset-path '../../../datasets/' --dataset miniimagenet --model resnet12 --batch-size 128 --epochs 0 --load-model '/ssd2/backbones/resnet12/miniimagenet/mini2.pt1' --n-augmentation 50 --save-augmented-features '/ssd2/data/AugmentedSamples/features/miniImagenet/AS300/miniAS300backbone21.pt' --seed 333333 ;

