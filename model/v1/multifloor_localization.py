{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "14c4ba06",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import torch\n",
    "import numpy as np\n",
    "import os"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "704d6954",
   "metadata": {},
   "outputs": [],
   "source": [
    "current_path = os.getcwd()\n",
    "# 向上返回两级目录\n",
    "print(current_path)\n",
    "parent_path = os.path.abspath(os.path.join(current_path, \"../..\"))\n",
    "print(parent_path)\n",
    "os.chdir(parent_path)\n",
    "print(os.getcwd())"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "MAML",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
