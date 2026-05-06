//! Phase 7: Declarative UI System — JSON DOM trees rendered into compositor windows.
//!
//! The LLM outputs `render_ui` and `update_ui` syscalls containing JSON that describes
//! UI element trees. This module parses them and renders to windows.

use alloc::string::String;
use alloc::vec::Vec;
use alloc::format;
use crate::framebuffer::{Framebuffer, colors, CHAR_WIDTH, LINE_HEIGHT};
use crate::compositor::Rect;

/// A node in the declarative UI tree.
#[derive(Debug, Clone)]
pub enum UiNode {
    /// A container holding children (like a div).
    Container {
        id: String,
        direction: LayoutDirection,
        children: Vec<UiNode>,
        padding: usize,
    },
    /// A text label.
    Text {
        id: String,
        content: String,
        color: u32,
        bold: bool,
    },
    /// A clickable button.
    Button {
        id: String,
        label: String,
        color: u32,
    },
    /// A text input field.
    Input {
        id: String,
        placeholder: String,
        value: String,
    },
    /// A list of items.
    List {
        id: String,
        items: Vec<String>,
    },
    /// A horizontal separator.
    Separator,
}

/// Layout direction for containers.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LayoutDirection {
    Vertical,
    Horizontal,
}

/// Semantic UI events generated from mouse/keyboard interaction.
#[derive(Debug, Clone)]
pub struct UiEvent {
    /// The ID of the element that was interacted with.
    pub target_id: String,
    /// The type of interaction.
    pub event_type: UiEventType,
}

#[derive(Debug, Clone)]
pub enum UiEventType {
    Click,
    Hover,
    Input(String),
}

impl UiEvent {
    /// Serialize to JSON for injection into LLM context.
    pub fn to_json(&self) -> String {
        match &self.event_type {
            UiEventType::Click => {
                format!(r#"{{"event": "ui_click", "target_id": "{}"}}"#, self.target_id)
            }
            UiEventType::Hover => {
                format!(r#"{{"event": "ui_hover", "target_id": "{}"}}"#, self.target_id)
            }
            UiEventType::Input(val) => {
                format!(
                    r#"{{"event": "ui_input", "target_id": "{}", "value": "{}"}}"#,
                    self.target_id, val
                )
            }
        }
    }
}

/// Renders a UI tree into a framebuffer region.
pub struct UiRenderer;

impl UiRenderer {
    /// Render a UI node tree into the given rectangular area.
    /// Returns the total height consumed.
    pub fn render(fb: &mut Framebuffer, node: &UiNode, area: Rect) -> usize {
        match node {
            UiNode::Container {
                children,
                direction,
                padding,
                ..
            } => {
                let mut offset = *padding;
                let inner = Rect {
                    x: area.x + *padding,
                    y: area.y + *padding,
                    width: area.width.saturating_sub(*padding * 2),
                    height: area.height.saturating_sub(*padding * 2),
                };

                for child in children {
                    let child_area = match direction {
                        LayoutDirection::Vertical => Rect {
                            x: inner.x,
                            y: inner.y + offset,
                            width: inner.width,
                            height: inner.height.saturating_sub(offset),
                        },
                        LayoutDirection::Horizontal => Rect {
                            x: inner.x + offset,
                            y: inner.y,
                            width: inner.width.saturating_sub(offset),
                            height: inner.height,
                        },
                    };

                    let consumed = Self::render(fb, child, child_area);
                    offset += consumed + 2;
                }

                offset + *padding
            }

            UiNode::Text {
                content, color, ..
            } => {
                fb.draw_string(area.x, area.y, content, *color);
                LINE_HEIGHT
            }

            UiNode::Button {
                label, color, ..
            } => {
                let btn_width = label.len() * CHAR_WIDTH + 16;
                let btn_height = LINE_HEIGHT + 8;

                fb.fill_rect(area.x, area.y, btn_width, btn_height, *color);
                fb.draw_rect(area.x, area.y, btn_width, btn_height, colors::WHITE);
                fb.draw_string(
                    area.x + 8,
                    area.y + 4,
                    label,
                    colors::WHITE,
                );

                btn_height
            }

            UiNode::Input {
                placeholder,
                value,
                ..
            } => {
                let input_width = area.width.min(300);
                let input_height = LINE_HEIGHT + 8;

                fb.fill_rect(area.x, area.y, input_width, input_height, 0x00222222);
                fb.draw_rect(area.x, area.y, input_width, input_height, colors::MID_GRAY);

                let display_text = if value.is_empty() { placeholder } else { value };
                let text_color = if value.is_empty() {
                    colors::MID_GRAY
                } else {
                    colors::WHITE
                };
                fb.draw_string(area.x + 4, area.y + 4, display_text, text_color);

                input_height
            }

            UiNode::List { items, .. } => {
                let mut y_offset = 0;
                for (i, item) in items.iter().enumerate() {
                    let bullet = if i < items.len() { "• " } else { "  " };
                    let text = format!("{}{}", bullet, item);
                    fb.draw_string(area.x, area.y + y_offset, &text, colors::DIM_WHITE);
                    y_offset += LINE_HEIGHT;
                }
                y_offset
            }

            UiNode::Separator => {
                for x in area.x..area.x + area.width {
                    fb.set_pixel(x, area.y + 2, colors::MID_GRAY);
                }
                5
            }
        }
    }

    /// Hit test: find which UI element ID is at position (x, y) within the tree.
    /// This is simplified — uses bounding boxes approximated from layout.
    pub fn hit_test(node: &UiNode, area: Rect, x: usize, y: usize) -> Option<String> {
        if !area.contains(x, y) {
            return None;
        }

        match node {
            UiNode::Button { id, label, .. } => {
                let btn_width = label.len() * CHAR_WIDTH + 16;
                let btn_height = LINE_HEIGHT + 8;
                let btn_rect = Rect {
                    x: area.x,
                    y: area.y,
                    width: btn_width,
                    height: btn_height,
                };
                if btn_rect.contains(x, y) {
                    return Some(id.clone());
                }
            }
            UiNode::Input { id, .. } => {
                let input_rect = Rect {
                    x: area.x,
                    y: area.y,
                    width: area.width.min(300),
                    height: LINE_HEIGHT + 8,
                };
                if input_rect.contains(x, y) {
                    return Some(id.clone());
                }
            }
            UiNode::Container {
                id,
                children,
                direction,
                padding,
            } => {
                let inner = Rect {
                    x: area.x + *padding,
                    y: area.y + *padding,
                    width: area.width.saturating_sub(*padding * 2),
                    height: area.height.saturating_sub(*padding * 2),
                };

                let mut offset = 0;
                for child in children {
                    let child_area = match direction {
                        LayoutDirection::Vertical => Rect {
                            x: inner.x,
                            y: inner.y + offset,
                            width: inner.width,
                            height: LINE_HEIGHT + 10,
                        },
                        LayoutDirection::Horizontal => Rect {
                            x: inner.x + offset,
                            y: inner.y,
                            width: 200,
                            height: inner.height,
                        },
                    };

                    if let Some(hit_id) = Self::hit_test(child, child_area, x, y) {
                        return Some(hit_id);
                    }
                    offset += LINE_HEIGHT + 10;
                }

                // Container itself was hit if no child matched
                if !id.is_empty() {
                    return Some(id.clone());
                }
            }
            _ => {}
        }

        None
    }
}
