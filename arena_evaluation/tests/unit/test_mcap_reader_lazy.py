import pathlib
import pytest
import tempfile
import polars as pl
from unittest import mock
from arena_evaluation.processing.mcap_reader import MCAPReader
from arena_evaluation.storage.schemas import TopicBundle


# Helper mock classes to simulate ROS2 decoders
class MockChannel:
    def __init__(self, topic, message_encoding="cdr"):
        self.topic = topic
        self.message_encoding = message_encoding


class MockMessage:
    def __init__(self, log_time, channel_id=0, data=None):
        self.log_time = log_time
        self.channel_id = channel_id
        self.data = data


def _identity_decoder_factory():
    """A decoder factory whose decoder returns message.data unchanged."""
    factory = mock.Mock()
    factory.decoder_for.return_value = lambda data: data
    return factory


class MockVector3:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = x
        self.y = y
        self.z = z


class MockQuaternion:
    def __init__(self, x=0.0, y=0.0, z=0.0, w=1.0):
        self.x = x
        self.y = y
        self.z = z
        self.w = w


class MockPose:
    def __init__(self, px=0.0, py=0.0, qx=0.0, qy=0.0, qz=0.0, qw=1.0):
        self.position = MockVector3(px, py)
        self.orientation = MockQuaternion(qx, qy, qz, qw)


class MockPoseWithCovariance:
    def __init__(self, px=0.0, py=0.0, qx=0.0, qy=0.0, qz=0.0, qw=1.0):
        self.pose = MockPose(px, py, qx, qy, qz, qw)


class MockTwist:
    def __init__(self, lx=0.0, az=0.0):
        self.linear = MockVector3(lx)
        self.angular = MockVector3(z=az)


class MockTwistWithCovariance:
    def __init__(self, lx=0.0, az=0.0):
        self.twist = MockTwist(lx, az)


class MockStamp:
    def __init__(self, sec=0, nanosec=0):
        self.sec = sec
        self.nanosec = nanosec


class MockHeader:
    def __init__(self, frame_id="", sec=0, nanosec=0):
        self.frame_id = frame_id
        self.stamp = MockStamp(sec, nanosec)


class MockOdomMsg:
    def __init__(self, px=0.0, py=0.0, lx=0.0, az=0.0, sec=0, nanosec=0):
        self.header = MockHeader("odom", sec, nanosec)
        self.pose = MockPoseWithCovariance(px, py)
        self.twist = MockTwistWithCovariance(lx, az)


def test_mcap_reader_lazy_chunking():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = pathlib.Path(tmpdir)
        mcap_file = tmp_path / "dummy.mcap"
        mcap_file.touch()

        # 10,005 messages to cross the 10,000-message chunk flush threshold
        mock_messages = []
        for i in range(10005):
            channel = MockChannel("/env_0/odom")
            ros_msg = MockOdomMsg(px=float(i), py=float(i * 2), lx=1.0, az=0.5)
            message = MockMessage(i * 1000, channel_id=0, data=ros_msg)
            mock_messages.append((None, channel, message))

        with mock.patch("arena_evaluation.processing.mcap_reader.NonSeekingReader") as mock_reader_cls:
            mock_reader_inst = mock_reader_cls.return_value
            mock_reader_inst.iter_messages.return_value = mock_messages
            mock_reader_inst._decoder_factories = [_identity_decoder_factory()]

            reader = MCAPReader(mcap_file)
            bundles = reader.read(tmp_path / "raw")
            bundle = bundles["env_0"]

            odom_parquet = tmp_path / "raw" / "env_0" / "odom.parquet"
            assert odom_parquet.exists()

            assert isinstance(bundle, TopicBundle)
            assert isinstance(bundle.odom, pl.LazyFrame)

            odom_df = bundle.odom.collect()
            assert len(odom_df) == 10005
            assert odom_df["pos_x"][0] == 0.0
            assert odom_df["pos_x"][10004] == 10004.0
            assert odom_df["pos_y"][10004] == 20008.0
            assert odom_df["time_ns"][10004] == 10004000


class MockTranslation:
    def __init__(self, x, y, z=0.0):
        self.x = x
        self.y = y
        self.z = z


class MockTransform:
    def __init__(self, tx, ty, qx=0.0, qy=0.0, qz=0.0, qw=1.0):
        self.translation = MockTranslation(tx, ty)
        self.rotation = MockQuaternion(qx, qy, qz, qw)


class MockStampedTransform:
    def __init__(self, frame_id, child_frame_id, tx, ty, qx=0.0, qy=0.0, qz=0.0, qw=1.0):
        self.header = MockHeader(frame_id)
        self.child_frame_id = child_frame_id
        self.transform = MockTransform(tx, ty, qx, qy, qz, qw)


class MockTFMsg:
    def __init__(self, transforms):
        self.transforms = transforms


def test_mcap_reader_tf_gt_extraction():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = pathlib.Path(tmpdir)
        mcap_file = tmp_path / "dummy.mcap"
        mcap_file.touch()

        # Create a mock /tf message that matches world-to-base_link ground truth conditions
        transforms = [MockStampedTransform(frame_id="map", child_frame_id="env_0/map", tx=0.0, ty=0.0, qw=1.0), MockStampedTransform(frame_id="map", child_frame_id="env_0/base_link", tx=5.5, ty=10.0, qw=1.0)]
        mock_messages = [(None, MockChannel("/tf"), MockMessage(999999, channel_id=0, data=MockTFMsg(transforms)))]

        with mock.patch("arena_evaluation.processing.mcap_reader.NonSeekingReader") as mock_reader_cls:
            mock_reader_inst = mock_reader_cls.return_value
            mock_reader_inst.iter_messages.return_value = mock_messages
            mock_reader_inst._decoder_factories = [_identity_decoder_factory()]

            reader = MCAPReader(mcap_file)
            bundles = reader.read(tmp_path / "raw")
            bundle = bundles["env_0"]

            tf_gt_parquet = tmp_path / "raw" / "env_0" / "tf_gt.parquet"
            assert tf_gt_parquet.exists()

            assert bundle.tf_gt is not None
            assert isinstance(bundle.tf_gt, pl.LazyFrame)

            tf_gt_df = bundle.tf_gt.collect()
            assert len(tf_gt_df) == 1
            assert tf_gt_df["time_ns"][0] == 999999
            assert tf_gt_df["pos_x_gt"][0] == 5.5
            assert tf_gt_df["pos_y_gt"][0] == 10.0
            assert tf_gt_df["yaw_gt"][0] == 0.0


def test_mcap_reader_env_offset_auto_detection():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = pathlib.Path(tmpdir)
        topics_dir = tmp_path / "topics"
        env_1_dir = topics_dir / "env_1"
        env_1_dir.mkdir(parents=True)

        # Create global static TF defining env_1 offset at (100.0, 50.0)
        tf_static_df = pl.DataFrame(
            {
                "time_ns": [1000],
                "frame_id": ["map"],
                "child_frame_id": ["env_1/map"],
                "trans_x": [100.0],
                "trans_y": [50.0],
                "trans_z": [0.0],
                "rot_x": [0.0],
                "rot_y": [0.0],
                "rot_z": [0.0],
                "rot_w": [1.0],
            }
        )
        tf_static_df.write_parquet(topics_dir / "tf_static.parquet")

        # tf_gt in global coordinates (105.0, 52.0) -> should be offset to (5.0, 2.0)
        tf_gt_df = pl.DataFrame(
            {
                "time_ns": [1000],
                "pos_x_gt": [105.0],
                "pos_y_gt": [52.0],
                "yaw_gt": [0.0],
            }
        )
        tf_gt_df.write_parquet(env_1_dir / "tf_gt.parquet")

        # odom in local coordinates (5.0, 2.0) -> should remain local at (5.0, 2.0)
        odom_df = pl.DataFrame(
            {
                "time_ns": [1000],
                "pos_x": [5.0],
                "pos_y": [2.0],
                "yaw": [0.0],
            }
        )
        odom_df.write_parquet(env_1_dir / "odom.parquet")

        reader = MCAPReader(tmp_path / "dummy.mcap")
        bundles = reader.load_bundles(topics_dir, tmp_path)
        bundle = bundles["env_1"]

        assert bundle.tf_gt is not None
        gt_res = bundle.tf_gt.collect()
        assert abs(gt_res["pos_x_gt"][0] - 5.0) < 1e-5
        assert abs(gt_res["pos_y_gt"][0] - 2.0) < 1e-5

        assert bundle.odom is not None
        odom_res = bundle.odom.collect()
        assert abs(odom_res["pos_x"][0] - 5.0) < 1e-5
        assert abs(odom_res["pos_y"][0] - 2.0) < 1e-5
