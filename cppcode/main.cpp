#include "mediapipe/framework/calculator_graph.h"
#include "mediapipe/framework/port/opencv_video_inc.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include <opencv2/opencv.hpp>
#include <iostream>

constexpr char kInputStream[] = "input_video";
constexpr char kOutputStream[] = "multi_hand_landmarks";

int main() {
    // Grafo embutido em string (não precisa de .pbtxt externo)
    std::string graph_config = R"pb(
        input_stream: "input_video"
        output_stream: "multi_hand_landmarks"
        node {
          calculator: "ImageFrameToGpuBufferCalculator"
          input_stream: "input_video"
          output_stream: "input_video_gpu"
        }
        node {
          calculator: "HandLandmarkTrackingGpu"
          input_stream: "IMAGE:input_video_gpu"
          output_stream: "LANDMARKS:multi_hand_landmarks"
        }
    )pb";

    mediapipe::CalculatorGraph graph;
    auto status = graph.Initialize(mediapipe::ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig>(graph_config));
    if (!status.ok()) { std::cerr << status.message(); return -1; }

    ASSIGN_OR_RETURN(auto poller, graph.AddOutputStreamPoller(kOutputStream));
    MP_RETURN_IF_ERROR(graph.StartRun({}));

    cv::VideoCapture cap(0);
    if (!cap.isOpened()) { std::cerr << "Erro ao abrir câmera\n"; return -1; }

    int frame_id = 0;
    cv::Mat frame;
    while (cap.read(frame)) {
        cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);

        auto input_frame = absl::make_unique<mediapipe::ImageFrame>(
            mediapipe::ImageFormat::SRGB, frame.cols, frame.rows,
            mediapipe::ImageFrame::kDefaultAlignmentBoundary);
        frame.copyTo(mediapipe::formats::MatView(input_frame.get()));

        MP_RETURN_IF_ERROR(graph.AddPacketToInputStream(
            kInputStream, mediapipe::Adopt(input_frame.release())
                          .At(mediapipe::Timestamp(frame_id++))));

        mediapipe::Packet packet;
        if (poller.Next(&packet)) {
            auto& landmarks = packet.Get<std::vector<mediapipe::NormalizedLandmarkList>>();
            for (const auto& hand : landmarks) {
                std::cout << "Mão detectada com " << hand.landmark_size() << " pontos\n";
            }
        }
    }
    MP_RETURN_IF_ERROR(graph.CloseInputStream(kInputStream));
    return graph.WaitUntilDone().ok() ? 0 : -1;
}
