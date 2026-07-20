import Testing
@testable import IndexTTSMenuBarCore

struct BackendFailureMessageTests {
    @Test func explainsHuggingFaceCacheAndNetworkFailure() {
        let message = BackendFailureMessage.classify(
            logTail: "LocalEntryNotFoundError: couldn't connect to 'https://huggingface.co'",
            exitCode: 1
        )
        #expect(message.contains("模型缓存不完整"))
        #expect(message.contains("Hugging Face"))
    }

    @Test func explainsOccupiedPort() {
        let message = BackendFailureMessage.classify(
            logTail: "OSError: [Errno 48] Address already in use",
            exitCode: 1
        )
        #expect(message.contains("端口"))
    }

    @Test func fallsBackToExitCode() {
        let message = BackendFailureMessage.classify(logTail: "unknown", exitCode: 9)
        #expect(message.contains("代码 9"))
    }
}
