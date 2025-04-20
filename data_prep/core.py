import glob
import os
import pdb
import pickle
import re
from collections import defaultdict
from typing import Callable, Dict, List
import posixpath

import paramiko

from homelab_rag.models.embeddings import LocalEmbedding, FlagEmbedding, Sent384


def parse_text_dataset(source: str, dest: str, embed_function: LocalEmbedding,
                       is_remote: bool, force: bool = False) -> None:
    """parse all text files in a given local directory, chunk, embed, and batch ship into vector store

    Args:
        source: source directory
        dest: local or server directory
        embed_function: function used to create similarity vector
        is_remote: whether or not the directory is on the linux server
        force: resaves over destination file if true, default false

    """
    # TODO make this work both remote and local...
    # open up connection to server
    txt_files = glob.glob(os.path.join(source, '*.txt'))
    if not is_remote:
        dest_files = glob.glob(os.path.join(source, '*.pkl'))
        dest_files = [os.path.splitext(i)[0] for i in dest_files]
    else:
        ssh_client = paramiko.SSHClient()
        ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh_client.connect(os.environ['LINUX_MACHINE'], username=os.environ['LINUX_USER'], password=os.environ['LINUX_PASS'])
        sftp = ssh_client.open_sftp()
        dest_files = sftp.listdir(dest)
        # filter out dirs
        dest_files = [i for i in dest_files if i.endswith("pkl")]
    if not force:
        txt_files = [os.path.splitext(i)[0] for i in txt_files if i not in dest_files]
    for file in txt_files:
        with open(file + '.txt', "r", errors='ignore') as f:
            contents = f.read()
        record: defaultdict = defaultdict(list)
        batched = chunk_terms_service(contents)
        name = os.path.splitext(os.path.basename(file))[0]
        for idx, item in enumerate(batched):
            record['name'].append(name)
            record['source_dir'].append(os.path.dirname(file))
            record['dest_dir'].append(os.path.join(dest, name + '.pkl'))
            record['source_length'].append(len(contents))
            record['source_batches'].append(len(batched))
            record['raw_text'].append(item)
            record['batch_id'].append(idx)
            record['embedding'].append(embed_function(item))
        print(f"Saving: {name}")
        if is_remote:
            # save tmp
            tmp_file = os.path.join(source, 'tmp.pkl')
            remote_file = posixpath.join(dest, name + '.pkl')
            with open(tmp_file, "wb") as out_file:
                pickle.dump(record, out_file)
            # move tmp
            sftp.put(tmp_file, remote_file)
        else:
            # save directly
            output_file = os.path.join(dest, name + '.pkl')
            with open(output_file, "wb") as out_file:
                pickle.dump(record, out_file)
    if is_remote:
        sftp.close()
        ssh_client.close()


def chunk_terms_service(doc_text: str, chunk_length: int = 1024) -> List[str]:
    # strip html if it happens to be in here
    # TODO maybe break this a bit cleaner to maintain entire sentences
    clean_text = re.sub(r"<.*?>", "", doc_text)
    clean_text = re.sub(r"\n+", "\n", clean_text)
    batched_results = [clean_text[i:i+chunk_length] for i in range(0, len(clean_text), chunk_length)]
    return batched_results


if __name__ == "__main__":
    source = r"F:\data\terms-service\text"
    dst = r"/data/text_datasets/terms_service"
    parse_text_dataset(source, dst, FlagEmbedding(), True)
