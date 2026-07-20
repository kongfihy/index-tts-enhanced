import AppKit
import CoreGraphics

@MainActor
enum StatusBarIconRenderer {
    private struct Palette {
        let core: NSColor
        let glowStrength: CGFloat
    }

    static func image(for status: BackendController.Status, activity: BackendController.Activity = .idle) -> NSImage {
        let size = NSSize(width: 22, height: 18)
        let image = NSImage(size: size, flipped: false) { rect in
            guard let context = NSGraphicsContext.current?.cgContext else { return false }
            context.setShouldAntialias(true)
            context.setAllowsAntialiasing(true)

            drawWaveform(in: context)
            drawStatusDot(in: context, status: status, activity: activity)
            return true
        }

        // Menu bar template images are forced to monochrome. This icon must keep
        // the small status accent in full color while the main glyph adapts to
        // the current appearance when it is rendered.
        image.isTemplate = false
        image.accessibilityDescription = "IndexTTS：\(status.title)"
        return image
    }

    private static func drawWaveform(in context: CGContext) {
        let bars: [(x: CGFloat, height: CGFloat)] = [
            (2.2, 4.2),
            (4.8, 7.4),
            (7.4, 12.0),
            (10.0, 8.4),
            (12.6, 5.2),
        ]
        let centerY: CGFloat = 8.5

        // The app's colorScheme does not reliably match the translucent menu
        // bar material. A dark outer silhouette plus a light inner core stays
        // legible on both bright and dark wallpapers without mode detection.
        drawBars(
            bars,
            centerY: centerY,
            width: 2.55,
            color: NSColor.black.withAlphaComponent(0.46),
            in: context
        )
        drawBars(
            bars,
            centerY: centerY,
            width: 1.42,
            color: NSColor.white.withAlphaComponent(0.96),
            in: context
        )
    }

    private static func drawBars(
        _ bars: [(x: CGFloat, height: CGFloat)],
        centerY: CGFloat,
        width: CGFloat,
        color: NSColor,
        in context: CGContext
    ) {
        context.saveGState()
        context.setFillColor(color.cgColor)
        for bar in bars {
            // Keep all layers optically centered while their widths differ.
            let baseWidth: CGFloat = 1.42
            let x = bar.x - (width - baseWidth) / 2
            let rect = CGRect(
                x: x,
                y: centerY - bar.height / 2,
                width: width,
                height: bar.height
            )
            context.addPath(CGPath(
                roundedRect: rect,
                cornerWidth: width / 2,
                cornerHeight: width / 2,
                transform: nil
            ))
        }
        context.fillPath()
        context.restoreGState()
    }

    private static func drawStatusDot(
        in context: CGContext,
        status: BackendController.Status,
        activity: BackendController.Activity
    ) {
        let palette = palette(for: status, activity: activity)
        let center = CGPoint(x: 15.8, y: 13.65)

        // Broad, quiet halo. The gradient is deliberately subtle so the icon
        // feels alive without looking like a notification demanding attention.
        let glowRadius: CGFloat = 5.2
        let glowColors = [
            palette.core.withAlphaComponent(0.34 * palette.glowStrength).cgColor,
            palette.core.withAlphaComponent(0.13 * palette.glowStrength).cgColor,
            palette.core.withAlphaComponent(0).cgColor,
        ] as CFArray
        let glowLocations: [CGFloat] = [0, 0.42, 1]

        if let gradient = CGGradient(
            colorsSpace: CGColorSpaceCreateDeviceRGB(),
            colors: glowColors,
            locations: glowLocations
        ) {
            context.saveGState()
            context.setBlendMode(.normal)
            context.drawRadialGradient(
                gradient,
                startCenter: center,
                startRadius: 0,
                endCenter: center,
                endRadius: glowRadius,
                options: [.drawsAfterEndLocation]
            )
            context.restoreGState()
        }

        // Dual separation rings use the same background-independent strategy
        // as the waveform: dark edge for light materials, light inset for dark.
        let outerRing = CGRect(x: center.x - 2.95, y: center.y - 2.95, width: 5.9, height: 5.9)
        context.saveGState()
        context.setFillColor(NSColor.black.withAlphaComponent(0.38).cgColor)
        context.fillEllipse(in: outerRing)
        context.restoreGState()

        let innerRing = CGRect(x: center.x - 2.55, y: center.y - 2.55, width: 5.1, height: 5.1)
        context.saveGState()
        context.setFillColor(NSColor.white.withAlphaComponent(0.76).cgColor)
        context.fillEllipse(in: innerRing)
        context.restoreGState()

        // Saturated core with a very small vertical gradient for depth.
        let coreRect = CGRect(x: center.x - 2.15, y: center.y - 2.15, width: 4.3, height: 4.3)
        let topColor = palette.core.blended(withFraction: 0.20, of: .white) ?? palette.core
        let bottomColor = palette.core.blended(withFraction: 0.12, of: .black) ?? palette.core
        let coreColors = [topColor.cgColor, bottomColor.cgColor] as CFArray
        let coreLocations: [CGFloat] = [0, 1]

        context.saveGState()
        context.addEllipse(in: coreRect)
        context.clip()
        if let gradient = CGGradient(
            colorsSpace: CGColorSpaceCreateDeviceRGB(),
            colors: coreColors,
            locations: coreLocations
        ) {
            context.drawLinearGradient(
                gradient,
                start: CGPoint(x: center.x, y: coreRect.maxY),
                end: CGPoint(x: center.x, y: coreRect.minY),
                options: []
            )
        } else {
            context.setFillColor(palette.core.cgColor)
            context.fill(coreRect)
        }
        context.restoreGState()

        // Tiny highlight: enough to suggest a polished surface, not a glossy orb.
        let highlightRect = CGRect(x: center.x - 1.15, y: center.y + 0.45, width: 1.15, height: 0.8)
        context.saveGState()
        context.setFillColor(NSColor.white.withAlphaComponent(0.54).cgColor)
        context.fillEllipse(in: highlightRect)
        context.restoreGState()
    }

    private static func palette(for status: BackendController.Status, activity: BackendController.Activity) -> Palette {
        if status == .running || status == .external {
            switch activity {
            case .generating:
                return Palette(core: NSColor(calibratedRed: 0.62, green: 0.49, blue: 1.00, alpha: 1), glowStrength: 0.96)
            case .queued:
                return Palette(core: NSColor(calibratedRed: 1.00, green: 0.72, blue: 0.28, alpha: 1), glowStrength: 0.86)
            case .taskFailed:
                return Palette(core: NSColor(calibratedRed: 1.00, green: 0.35, blue: 0.42, alpha: 1), glowStrength: 0.90)
            case .idle:
                break
            }
        }
        switch status {
        case .stopped:
            return Palette(
                core: NSColor(calibratedRed: 0.57, green: 0.60, blue: 0.66, alpha: 1),
                glowStrength: 0.42
            )
        case .starting, .stopping:
            return Palette(
                core: NSColor(calibratedRed: 1.00, green: 0.72, blue: 0.28, alpha: 1),
                glowStrength: 0.84
            )
        case .running:
            return Palette(
                core: NSColor(calibratedRed: 0.31, green: 0.84, blue: 0.55, alpha: 1),
                glowStrength: 0.78
            )
        case .external:
            return Palette(
                core: NSColor(calibratedRed: 0.36, green: 0.64, blue: 1.00, alpha: 1),
                glowStrength: 0.72
            )
        case .failed:
            return Palette(
                core: NSColor(calibratedRed: 1.00, green: 0.35, blue: 0.42, alpha: 1),
                glowStrength: 0.90
            )
        }
    }
}
