#!/usr/bin/env python3
# -*-coding: utf-8 -*-
# pylint: disable=invalid-name,no-member

def dice(output, target, e=1e-9):
    i_flat = output.contiguous().view(-1).clamp(min=0)
    t_flat = target.contiguous().view(-1)
    _correct = 2 * (i_flat * t_flat).sum() + e
    _examples = ((i_flat * i_flat).sum() + (t_flat * t_flat).sum()) + e
    return (_correct / _examples).to("cpu").tolist()[0]
