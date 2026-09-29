#!/usr/bin/env python3
from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path


def varray(root: ET.Element, name: str) -> list[list[float]]:
    node = root.find(f".//varray[@name='{name}']")
    if node is None:
        raise ValueError(f"Could not find varray name={name!r}")
    rows = []
    for v in node.findall("v"):
        rows.append([float(x) for x in v.text.split()])
    return rows


def born_charges(root: ET.Element) -> list[list[list[float]]]:
    node = root.find(".//array[@name='born_charges']")
    if node is None:
        raise ValueError("Could not find array name='born_charges'")
    tensors = []
    for rc in node.findall(".//set[@comment='ion']/set[@comment='born_charges']"):
        tensor = []
        for v in rc.findall("v"):
            tensor.append([float(x) for x in v.text.split()])
        if tensor:
            tensors.append(tensor)
    if not tensors:
        for ion_set in node.findall("./set"):
            tensor = [[float(x) for x in v.text.split()] for v in ion_set.findall("v")]
            if len(tensor) == 3:
                tensors.append(tensor)
    if not tensors:
        for ion_set in node.findall(".//set/set"):
            tensor = [[float(x) for x in v.text.split()] for v in ion_set.findall("v")]
            if len(tensor) == 3:
                tensors.append(tensor)
    if not tensors:
        raise ValueError("Could not parse Born charge tensors from vasprun.xml")
    return tensors


def main():
    parser = argparse.ArgumentParser(description="Convert VASP LEPSILON vasprun.xml to BORNINFO.")
    parser.add_argument("vasprun", type=Path)
    parser.add_argument("-o", "--output", type=Path, default=Path("BORNINFO"))
    args = parser.parse_args()

    root = ET.parse(args.vasprun).getroot()
    eps = varray(root, "epsilon")
    born = born_charges(root)

    with args.output.open("w", encoding="utf-8") as fh:
        for row in eps:
            fh.write(" ".join(f"{x:16.8f}" for x in row) + "\n")
        for tensor in born:
            for row in tensor:
                fh.write(" ".join(f"{x:16.8f}" for x in row) + "\n")
    print(f"[DONE] wrote {args.output} with {len(born)} Born tensors")


if __name__ == "__main__":
    main()
