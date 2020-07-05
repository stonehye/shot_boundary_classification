video_list=("beautiful_mind_game_theory/5f6527456bf2b4c460737a1c222dbd0e70009030.flv" "beautiful_mind_game_theory/b41c61fb756014444b7974ef25f2c76d5c3b42e4.flv" "baggio_penalty_1994/6d1a89c83d554fc6a5e39fcadb172a79baf140fd.mp4" "beckham_70_yard_goal/1f4f6311bb4acc0bc72e4a915a32093a7a85741c.flv" "president_obama_takes_oath/00274a923e13506819bd273c694d10cfa07ce1ec.flv" "beautiful_mind_game_theory/46f2e964ae16f5c27fad70d6849c76616fad7502.flv" "beautiful_mind_game_theory/6171d3d87ae377e497199554033bca96a263277b.mp4" "beautiful_mind_game_theory/904c8ebf782357ae78ebd205fe3428ad76b975a5.flv" "baggio_penalty_1994/3504e360accbaccb1580befbb441f1019664c2bb.mp4" "baggio_penalty_1994/bb604f57a18455867544e79c2e32bf5583c358d4.flv" "beckham_70_yard_goal/3fe2e54fdb267849aa8aeb0b39ec0d1c7075b391.flv" "beckham_70_yard_goal/4ffe88b2da93f727952dc3104e70c0b0665d9e1a.flv" "beckham_70_yard_goal/66e0225621e48946b7ab09027c18c83b69564002.flv" "beckham_70_yard_goal/1357538fd1fd2975a8b3f750d5f6b13695aa4e28.flv" "beckham_70_yard_goal/37a935b4c21662e183a0dd632f40d5fd6900c42b.mp4" "president_obama_takes_oath/66246823d79897fd7d93677c6c0244a3cbd4657f.flv")
dir="/hdd/stonehye/VCDB/core_dataset/videos/"

for i in ${video_list[@]};
do
    python3 inference.py --option Video --videopath $dir$i
    # echo $dir$i
done
