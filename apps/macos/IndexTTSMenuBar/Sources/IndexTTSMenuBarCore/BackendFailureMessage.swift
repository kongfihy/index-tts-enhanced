import Foundation

public enum BackendFailureMessage {
    public static func classify(logTail: String, exitCode: Int32) -> String {
        let log = logTail.lowercased()

        if log.contains("couldn't connect to 'https://huggingface.co'")
            || log.contains("localentrynotfounderror")
            || log.contains("hugging face download failed")
        {
            return "模型缓存不完整，且无法连接 Hugging Face。请检查网络后重试，或打开日志查看缺少的模型。"
        }

        if log.contains("address already in use") || log.contains("errno 48") {
            return "启动端口已被其他程序占用。请检查 7860/7861 端口，或打开日志查看详情。"
        }

        if log.contains("required file") && log.contains("does not exist") {
            return "本地模型文件不完整。请打开日志查看具体缺少的文件。"
        }

        if log.contains("unable to load") && log.contains("local caches were checked first") {
            return "本地缓存中没有找到完整模型，联网补全也失败了。请检查网络或打开日志。"
        }

        return "后端异常退出（代码 \(exitCode)），请打开日志查看详情。"
    }
}
