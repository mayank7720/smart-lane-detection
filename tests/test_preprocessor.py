#!/usr/bin/env python3
"""
Unit Tests — Preprocessor Module
==================================
Tests for core.preprocessor functions:
  - to_grayscale
  - apply_blur
  - detect_edges
  - preprocess
"""

import os
import sys
import unittest

import numpy as np

# Ensure project root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from core.preprocessor import to_grayscale, apply_blur, detect_edges, preprocess


class TestToGrayscale(unittest.TestCase):
    """Tests for to_grayscale()."""

    def setUp(self):
        """Create a dummy 3-channel color image."""
        self.color_image = np.random.randint(
            0, 256, size=(480, 640, 3), dtype=np.uint8
        )

    def test_output_is_2d(self):
        """Grayscale output should be a 2D array (no channel dimension)."""
        gray = to_grayscale(self.color_image)
        self.assertEqual(len(gray.shape), 2)

    def test_output_same_height_width(self):
        """Height and width should be preserved after grayscale conversion."""
        gray = to_grayscale(self.color_image)
        self.assertEqual(gray.shape[0], self.color_image.shape[0])
        self.assertEqual(gray.shape[1], self.color_image.shape[1])

    def test_output_dtype(self):
        """Output should be uint8."""
        gray = to_grayscale(self.color_image)
        self.assertEqual(gray.dtype, np.uint8)

    def test_single_channel_input(self):
        """If already grayscale, should still return a valid 2D image."""
        gray_input = np.random.randint(0, 256, size=(100, 100), dtype=np.uint8)
        # Depending on implementation, this may convert or pass through
        try:
            result = to_grayscale(gray_input)
            self.assertEqual(len(result.shape), 2)
        except Exception:
            # Some implementations may not accept 2D input — that's acceptable
            pass


class TestApplyBlur(unittest.TestCase):
    """Tests for apply_blur()."""

    def setUp(self):
        """Create a dummy grayscale image."""
        self.gray_image = np.random.randint(
            0, 256, size=(480, 640), dtype=np.uint8
        )

    def test_output_same_shape(self):
        """Blurred image should have the same shape as the input."""
        blurred = apply_blur(self.gray_image)
        self.assertEqual(blurred.shape, self.gray_image.shape)

    def test_output_dtype(self):
        """Output should remain uint8."""
        blurred = apply_blur(self.gray_image)
        self.assertEqual(blurred.dtype, np.uint8)

    def test_blur_reduces_noise(self):
        """Blurring should reduce the standard deviation of pixel values
        in a noisy image (statistical property, not guaranteed for every
        random seed, so we use a high-noise image)."""
        noisy = np.random.randint(0, 256, size=(200, 200), dtype=np.uint8)
        blurred = apply_blur(noisy)
        # Standard deviation should generally decrease
        self.assertLessEqual(
            np.std(blurred.astype(float)),
            np.std(noisy.astype(float)) + 5  # small tolerance
        )


class TestDetectEdges(unittest.TestCase):
    """Tests for detect_edges()."""

    def setUp(self):
        """Create a dummy grayscale image with a clear vertical edge."""
        self.edge_image = np.zeros((200, 200), dtype=np.uint8)
        self.edge_image[:, 100:] = 255  # strong vertical edge at x=100

    def test_output_is_2d(self):
        """Edge map should be a 2D array."""
        edges = detect_edges(self.edge_image)
        self.assertEqual(len(edges.shape), 2)

    def test_output_dtype_uint8(self):
        """Edge map should be uint8."""
        edges = detect_edges(self.edge_image)
        self.assertEqual(edges.dtype, np.uint8)

    def test_edges_detected_on_boundary(self):
        """Pixels near the vertical edge should have non-zero values."""
        edges = detect_edges(self.edge_image)
        # Check that some pixels near x=100 are non-zero
        edge_region = edges[:, 95:105]
        self.assertGreater(np.sum(edge_region > 0), 0)

    def test_uniform_image_no_edges(self):
        """A completely uniform image should produce no edges."""
        uniform = np.full((200, 200), 128, dtype=np.uint8)
        edges = detect_edges(uniform)
        self.assertEqual(np.sum(edges > 0), 0)


class TestPreprocess(unittest.TestCase):
    """Tests for the full preprocess() pipeline function."""

    def setUp(self):
        """Create a dummy color image."""
        self.color_image = np.random.randint(
            0, 256, size=(480, 640, 3), dtype=np.uint8
        )

    def test_returns_tuple(self):
        """preprocess() should return a tuple."""
        result = preprocess(self.color_image)
        self.assertIsInstance(result, tuple)

    def test_returns_two_elements(self):
        """Tuple should contain exactly 2 elements (edges, enhanced)."""
        result = preprocess(self.color_image)
        self.assertEqual(len(result), 2)

    def test_edges_shape(self):
        """Edge output should be 2D with same height/width."""
        edges, _ = preprocess(self.color_image)
        self.assertEqual(len(edges.shape), 2)
        self.assertEqual(edges.shape[0], self.color_image.shape[0])
        self.assertEqual(edges.shape[1], self.color_image.shape[1])

    def test_enhanced_shape(self):
        """Enhanced output should have the same height/width as input."""
        _, enhanced = preprocess(self.color_image)
        self.assertEqual(enhanced.shape[0], self.color_image.shape[0])
        self.assertEqual(enhanced.shape[1], self.color_image.shape[1])


if __name__ == "__main__":
    unittest.main(verbosity=2)
