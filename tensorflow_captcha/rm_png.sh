while true ; 
do
sleep 10;
find . -name "*.png" | xargs rm -r
echo 'rm done';
done
