"""End-to-end integration test for perception pipeline without HTTP layer."""
import io
import unittest
import numpy as np
from PIL import Image

from backend.app.api.validation import validate_image_upload
from backend.app.vision.image_quality import quality_analyzer
from backend.app.vision.preprocessing import preprocessor
from backend.app.vision.detector import road_detector
from backend.app.vision.ocr import ocr_service
from backend.app.vision.postprocessing import postprocessor
from backend.app.intelligence.interpreter import interpreter
from backend.app.performance.benchmark import PipelineProfiler
from backend.app.schemas import (
    AnalyzeResponse,
    ImageDimensions,
    ImageQualityDetails,
    PerformanceDetails,
)


class TestPipelineEndToEnd(unittest.TestCase):
    """Verifies complete stage integration from upload to structured road intelligence."""

    def test_full_pipeline_execution(self):
        # 1. Synthesize road scene image
        raw_img = Image.fromarray(np.random.randint(0, 255, (200, 300, 3), dtype=np.uint8))
        buf = io.BytesIO()
        raw_img.save(buf, format="JPEG")
        content = buf.getvalue()

        # 2. Upload validation
        pil_img, fmt = validate_image_upload("highway.jpg", content, "image/jpeg")
        self.assertEqual(fmt, "jpeg")

        img_rgb = np.array(pil_img)
        img_bgr = img_rgb[:, :, ::-1].copy()
        h, w, c = img_bgr.shape
        self.assertEqual((h, w, c), (200, 300, 3))

        # 3. Profiling & Quality Analysis
        profiler = PipelineProfiler()
        profiler.start_pipeline()

        profiler.start_stage("quality_check")
        quality = quality_analyzer.analyze(img_bgr)
        profiler.end_stage("quality_check")
        self.assertIn("blur_score", quality)

        # 4. Adaptive Preprocessing
        profiler.start_stage("preprocessing")
        proc_img, applied_transforms = preprocessor.adaptively_preprocess(img_bgr, quality)
        profiler.end_stage("preprocessing")
        self.assertEqual(proc_img.shape, img_bgr.shape)

        # 5. Detection & Region Extraction
        profiler.start_stage("detection")
        detections = road_detector.detect(proc_img)
        profiler.end_stage("detection")
        self.assertIsInstance(detections, list)

        # 6. OCR Extraction & Postprocessing
        profiler.start_stage("ocr")
        candidate_boxes = [d.bounding_box for d in detections]
        raw_ocr = ocr_service.extract_from_regions(proc_img, candidate_boxes)
        ocr_results = postprocessor.postprocess_ocr(raw_ocr, max_width=w, max_height=h)
        profiler.end_stage("ocr")
        self.assertIsInstance(ocr_results, list)

        # 7. Semantic Understanding
        profiler.start_stage("intelligence")
        intelligence = interpreter.interpret(detections, ocr_results, quality)
        profiler.end_stage("intelligence")
        self.assertIsNotNone(intelligence.object_type)

        # 8. Performance Finish
        perf = profiler.finish_pipeline()
        self.assertIn("total", perf["latency_ms"])
        self.assertGreater(perf["latency_ms"]["total"], 0.0)

        # 9. Schema Assembly
        resp = AnalyzeResponse(
            success=True,
            filename="highway.jpg",
            image_dimensions=ImageDimensions(width=w, height=h, channels=c),
            detections=detections,
            ocr_results=ocr_results,
            road_intelligence=intelligence,
            image_quality=ImageQualityDetails(**quality),
            performance=PerformanceDetails(**perf),
        )

        resp_dict = resp.model_dump()
        self.assertTrue(resp_dict["success"])
        self.assertEqual(resp_dict["filename"], "highway.jpg")
        self.assertIn("performance", resp_dict)
        self.assertIn("road_intelligence", resp_dict)


if __name__ == "__main__":
    unittest.main()
