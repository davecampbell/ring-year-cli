# ring-year-cli

## operation

build the image from the Dockerfile  

this command will run the container in 'look' mode:  
- assumes the image name is 'ring-year-cli'  
- it 'looks' in code/images/look/ folder for an image  
  
docker run -v$(pwd):/app ring-year-cli bash -c "cd code && python predict.py -m look"

