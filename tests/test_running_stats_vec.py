import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
import numpy as np
import pytest

from src.running_stats_vec import RunningStatsVec


def test_init():
    rs = RunningStatsVec()
    assert np.allclose(rs.mean, 0.0)
    assert np.allclose(rs.variance, 0.0)


def test_constant():
    rs = RunningStatsVec()
    data = np.random.randn(100, 10000)
    rs.push(data)
    assert np.allclose(rs.mean, data.mean(axis=1, keepdims=True))
    assert np.allclose(rs.variance, data.var(axis=1, keepdims=True))


@pytest.mark.parametrize("num_samples", [1, 2, 3, 4, 5])
def test_constant_aggregate(num_samples):
    rs = RunningStatsVec()
    all_data = []
    for _ in range(num_samples):
        data = np.random.randn(100, 10000)
        rs.push(data)
        all_data.append(data)
    all_data = np.concatenate(all_data, axis=1)
    assert np.allclose(rs.mean, all_data.mean(axis=1, keepdims=True))
    assert np.allclose(rs.variance, all_data.var(axis=1, keepdims=True))


def test_random():
    rs = RunningStatsVec()
    data = np.random.randn(100, 10000)
    data = (data + np.arange(100).reshape(-1, 1)) * np.random.rand(100, 1) * 10
    rs.push(data)
    assert np.allclose(rs.mean, data.mean(axis=1, keepdims=True))
    assert np.allclose(rs.variance, data.var(axis=1, keepdims=True))


@pytest.mark.parametrize("num_samples", [1, 2, 3, 4, 5])
def test_random_aggregate(num_samples):
    rs = RunningStatsVec()
    all_data = []
    for _ in range(num_samples):
        data = np.random.randn(100, 10000)
        data = (data + np.random.rand(100, 1) * 100) * np.random.rand(100, 1) * 10
        rs.push(data)
        all_data.append(data)
    all_data = np.concatenate(all_data, axis=1)
    assert np.allclose(rs.mean, all_data.mean(axis=1, keepdims=True))
    assert np.allclose(rs.variance, all_data.var(axis=1, keepdims=True))


def test_clear():
    rs = RunningStatsVec()
    data = np.random.randn(100, 10000)
    rs.push(data)
    rs.clear()
    assert np.allclose(rs.mean, 0.0)
    assert np.allclose(rs.variance, 0.0)
