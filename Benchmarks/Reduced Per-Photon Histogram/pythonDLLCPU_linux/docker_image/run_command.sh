#!/bin/bash
echo "running"
mount -t cifs //129.2.116.114/rbry /NAS -o credentials=/home/rbry/auth/NAS_auth
if [ -n "$1" ]
then
        if [ "$1" == "command" ]
        then
                echo "Processing..."
                python /dummy_processor.py "$@"
        elif [ "$1" == "file" ]
        then
                echo "Running file"
        elif [ "$1" == "jupyter" ]
        then
                jupyter contrib nbextension install --user
                jupyter nbextension enable scroll_down/main --user
                jupyter notebook --ip=0.0.0.0 --no-browser --allow-root
        fi
else
        echo "Nothing to do"
fi
