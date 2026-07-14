"""Test định tuyến intent + SAFETY override của orchestrator (stub skills, synthetic)."""

from app.orchestrator import VisionAssistant


def new():
    return VisionAssistant(embedder_kind="synthetic")


def test_teach_and_recognize_face():
    a = new()
    a.handle("Hãy nhớ đây là Lan", frame="p_lan")
    r = a.handle("Ai đây?", frame="p_lan")
    assert "Lan" in r


def test_object_modality_routing():
    a = new()
    a.handle("Hãy nhớ đây là ví của tôi", frame="wallet")
    r = a.handle("Cái gì đây?", frame="wallet")
    assert "ví của tôi" in r


def test_scene_default():
    a = new()
    r = a.handle("Trước mặt có gì?")
    assert "phía trước" in r.lower()


def test_ocr_routing():
    a = new()
    r = a.handle("Đọc chữ trên biển giúp tôi")
    assert "chữ đọc được" in r.lower()


def test_safety_override_blocks_scene_skill():
    a = new()
    a.obstacle.set_distance(0.8)  # nguy hiểm
    r = a.handle("Trước mặt có gì?")
    assert r.startswith("Cảnh báo")
    assert "phía trước là" not in r.lower()


def test_safety_override_blocks_teach_and_recognition():
    a = new()
    a.obstacle.set_distance(0.8)

    assert "Cảnh báo" in a.handle("Hãy nhớ đây là Lan", frame="p_lan")
    assert a.core.cpm["face"].labels() == []
    assert "Cảnh báo" in a.handle("Ai đây?", frame="p_lan")


def test_obstacle_query():
    a = new()
    a.obstacle.set_distance(0.8)
    assert "vật cản" in a.handle("Đường có an toàn không?").lower()
    a.obstacle.set_distance(None)
    assert "thông thoáng" in a.handle("Có vật cản gì không?").lower()


def test_correction():
    a = new()
    a.handle("Sửa, đây là Huy", frame="p_x")
    assert "Huy" in a.handle("Ai đây?", frame="p_x")
