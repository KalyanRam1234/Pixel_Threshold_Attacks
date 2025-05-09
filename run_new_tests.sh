# for attack in pixel threshold 
# do
#     for es in 1 0
#     do
#         for family in 3
#         do
#             for dataset in 0
#             do
#                 for model in 1
#                 do
#                     python -u code/run_attack.py $attack -e $es -f $family -d $dataset -m $model --samples 1 --epochs 10 -th 64 --verbose --plot_image --batch_size 8
#                 done
#             done
#         done
#     done
# done

## Cifar Yolo training

# for attack in pixel threshold 
# do
#     for es in 1 0
#     do
#         for family in 1
#         do
#             for dataset in 0
#             do
#                 for model in 10
#                 do
#                     python -u code/run_attack.py $attack -e $es -f $family -d $dataset -m $model --samples 1 --epochs 20 -th 16 --verbose --plot_image --batch_size 8
#                 done
#             done
#         done
#     done
# done


## Imagenet Classification

# for attack in pixel threshold 
# do
#     for es in 1 0
#     do
#         for family in 3
#         do
#             for dataset in 0
#             do
#                 for model in 2
#                 do
#                     python -u code/run_attack.py $attack -e $es -f $family -d $dataset -m $model --samples 1 --epochs 50 -th 16 --verbose --plot_image --batch_size 2
#                 done
#             done
#         done
#     done
# done

## Xview Dataset
for attack in pixel threshold 
do
    for es in 1 0
    do
        for family in 4
        do
            for dataset in 0
            do
                for model in 1
                do
                    python -u code/run_attack.py $attack -e $es -f $family -d $dataset -m $model --samples 1 --epochs 20 -th 32 --verbose --plot_image --batch_size 4 --samples 3
                done
            done
        done
    done
done

## Cifar Dataset

# for attack in pixel threshold 
# do
#     for es in 1 0
#     do
#         for family in 1
#         do
#             for dataset in 0
#             do
#                 for model in 1
#                 do
#                     python -u code/run_attack.py $attack -e $es -f $family -d $dataset -m $model --samples 1 --epochs 10 -th 16 -v --plot_image --batch_size 2
#                 done
#             done
#         done
#     done
# done