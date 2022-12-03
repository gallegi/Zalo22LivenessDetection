# python3 train.py --data
cd /code

python3 predict_ens.py --weight models/v5_seg_head_regnet_y_16gf/fold0/best.pth \
                        --weight_seq models/cspdarknet_lstm/fold0/best.pt \
                        --test_video_dir /data/private_test/videos \
                        --submission_folder /result \
                        --output_name submission