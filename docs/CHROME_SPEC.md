# Chrome Bookmarks File Format Specification

## Overview

This document provides a comprehensive specification of the Chrome bookmarks export file format, based on analysis of exported bookmark files and the Chrome bookmarks interface (`chrome://bookmarks/`). The format is based on the Netscape bookmark file format, which has become a standard for bookmark interchange between browsers.

## Chrome Bookmarks Interface Structure

The Chrome bookmarks manager (`chrome://bookmarks/`) organizes bookmarks in a hierarchical structure with the following key components:

### Interface Layout
- **Left Panel**: Tree-based folder navigation with expandable/collapsible folders
- **Right Panel**: Grid/list view showing contents of the selected folder
- **Top Bar**: Search functionality and organize menu
- **URL Pattern**: `chrome://bookmarks/?id={folder_id}` where each folder has a unique numeric ID

### Default Folder Structure
1. **Bookmarks Bar** (ID: 1) - `PERSONAL_TOOLBAR_FOLDER="true"` in HTML export
2. **Other Bookmarks** (ID: 2) - Default location for non-toolbar bookmarks
3. **Mobile Bookmarks** (ID: 3) - If Chrome sync is enabled (may not appear in all exports)

### Folder Navigation
- Folders can be expanded/collapsed in the tree view
- Clicking a folder updates the URL to `chrome://bookmarks/?id={folder_id}`
- Subfolders are visually nested in the tree hierarchy
- Empty folders are displayed but show "0 items in bookmark list"

## File Structure

### Basic Format
- **File Type**: HTML document
- **Character Encoding**: UTF-8
- **MIME Type**: `text/html`
- **File Extension**: `.html`

### Document Declaration
```html
<!DOCTYPE NETSCAPE-Bookmark-file-1>
<!-- This is an automatically generated file.
     It will be read and overwritten.
     DO NOT EDIT! -->
<META HTTP-EQUIV="Content-Type" CONTENT="text/html; charset=UTF-8">
<TITLE>Bookmarks</TITLE>
<H1>Bookmarks</H1>
```

#### Key Components:
1. **DOCTYPE**: `NETSCAPE-Bookmark-file-1` - Legacy format identifier
2. **Warning Comment**: Indicates the file is auto-generated
3. **Meta Tags**: Specify UTF-8 encoding
4. **Title**: Document title "Bookmarks"
5. **H1**: Top-level heading "Bookmarks"

## Hierarchical Structure

### Container Elements

#### Definition List (`<DL>`)
- **Purpose**: Container for bookmark folders and items
- **Usage**: Wraps groups of bookmarks and folders
- **Nesting**: Can be nested for folder hierarchies
- **Format**: `<DL><p>` (note the paragraph tag)

#### Definition Term (`<DT>`)
- **Purpose**: Wrapper for individual bookmark entries
- **Usage**: Contains either folder headers (`<H3>`) or bookmark links (`<A>`)
- **Required**: Every bookmark or folder must be wrapped in `<DT>`

### Folder Structure

#### Folder Header (`<H3>`)
```html
<DT><H3 ADD_DATE="1634868593" LAST_MODIFIED="1635453203" PERSONAL_TOOLBAR_FOLDER="true">Folder Name</H3>
```

**Attributes:**
- `ADD_DATE`: Unix timestamp when folder was created
- `LAST_MODIFIED`: Unix timestamp when folder was last modified
- `PERSONAL_TOOLBAR_FOLDER`: Boolean ("true" only for Bookmarks Bar)

**Special Folders:**
1. **Bookmarks Bar**: `PERSONAL_TOOLBAR_FOLDER="true"` - Corresponds to Chrome UI folder ID 1
2. **Other Bookmarks**: No special attribute - Corresponds to Chrome UI folder ID 2
3. **Mobile Bookmarks**: No special attribute (if present) - Corresponds to Chrome UI folder ID 3

**Folder ID Mapping:**
- The HTML export format does not include internal folder IDs
- Chrome's internal ID system (visible in `chrome://bookmarks/?id=N`) is not preserved in exports
- When importing, Chrome assigns new internal IDs based on the hierarchical structure

### Bookmark Links

#### Link Format (`<A>`)
```html
<DT><A HREF="https://example.com/" ADD_DATE="1234567890" ICON="data:image/png;base64,...">Link Title</A>
```

**Required Attributes:**
- `HREF`: The URL of the bookmark

**Optional Attributes:**
- `ADD_DATE`: Unix timestamp when bookmark was added
- `ICON`: Base64-encoded favicon data URL
- `LAST_VISIT`: Unix timestamp of last visit (rare)
- `LAST_MODIFIED`: Unix timestamp when bookmark was modified (rare)

#### Favicon Format
```
ICON="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9h..."
```
- **Format**: Data URL with base64 encoding
- **Image Type**: Usually PNG, sometimes ICO or JPEG
- **Size**: Typically 16x16 pixels
- **Purpose**: Display icon next to bookmark title

## Complete Structure Example

```html
<!DOCTYPE NETSCAPE-Bookmark-file-1>
<!-- This is an automatically generated file.
     It will be read and overwritten.
     DO NOT EDIT! -->
<META HTTP-EQUIV="Content-Type" CONTENT="text/html; charset=UTF-8">
<TITLE>Bookmarks</TITLE>
<H1>Bookmarks</H1>
<DL><p>
    <DT><H3 ADD_DATE="1715434444" LAST_MODIFIED="1717526901" PERSONAL_TOOLBAR_FOLDER="true">Bookmarks bar</H3>
    <DL><p>
        <DT><H3 ADD_DATE="1717175257" LAST_MODIFIED="1717526910">Machine Learning</H3>
        <DL><p>
            <DT><A HREF="https://example.com/" ADD_DATE="1717175221" ICON="data:image/png;base64,...">Example Site</A>
            <DT><A HREF="https://another-site.com/" ADD_DATE="1717175261">Another Site</A>
        </DL><p>
        <DT><A HREF="https://direct-bookmark.com/" ADD_DATE="1717175273">Direct Bookmark</A>
    </DL><p>
    <DT><H3 ADD_DATE="1634868593" LAST_MODIFIED="1634868593">Other Folder</H3>
    <DL><p>
        <DT><A HREF="https://nested-bookmark.com/" ADD_DATE="1634868593">Nested Bookmark</A>
    </DL><p>
</DL><p>
```

## Data Types and Formats

### Timestamps
- **Format**: Unix timestamp (seconds since January 1, 1970)
- **Type**: Integer string
- **Example**: `"1717175221"` represents 2024-05-31 15:07:01 UTC
- **Usage**: ADD_DATE, LAST_MODIFIED, LAST_VISIT

### URLs
- **Format**: Full URL including protocol
- **Encoding**: URL-encoded if necessary
- **Examples**: 
  - `https://example.com/`
  - `http://localhost:3000/`
  - `file:///C:/Users/User/file.pdf`

### Icons
- **Format**: Data URL with base64 encoding
- **Schema**: `data:image/{type};base64,{base64-data}`
- **Image Types**: png, jpeg, ico, gif, webp
- **Size**: Usually 16x16 pixels
- **Optional**: May be omitted if no favicon available

### Text Content
- **Encoding**: UTF-8
- **HTML Entities**: Standard HTML entities used when necessary
- **Examples**: `&amp;`, `&lt;`, `&gt;`, `&#39;`

## Nesting Rules

### Depth Limitations
- **Maximum Depth**: No explicit limit, but practically limited by browser UI
- **Common Depth**: 3-5 levels typical for most users
- **Performance**: Deep nesting may impact parsing performance

### Folder Relationships
- **Parent-Child**: Folders can contain other folders and bookmarks
- **Siblings**: Folders and bookmarks can exist at the same level
- **Order**: Items appear in the order they're defined in the file

## Special Characteristics

### Bookmarks Bar
- **Identifier**: `PERSONAL_TOOLBAR_FOLDER="true"`
- **Visibility**: Typically shown in browser toolbar
- **Limit**: Usually one per bookmark file (the first one)

### File URLs
- **Support**: Local file references supported
- **Format**: `file:///absolute/path/to/file`
- **Platform**: Path format depends on operating system

### Empty Folders
- **Allowed**: Folders can be empty
- **Format**: `<DL><p></DL><p>` or completely omitted
- **Display**: May or may not be shown in browser depending on settings

## Parsing Considerations

### Error Handling
- **Malformed HTML**: Browsers typically handle gracefully
- **Missing Attributes**: Optional attributes can be omitted
- **Invalid URLs**: Invalid URLs may be preserved but marked as broken
- **Character Encoding**: Must handle UTF-8 properly

### Performance
- **File Size**: Large bookmark files (>10MB) may cause performance issues
- **Icon Data**: Base64 favicon data significantly increases file size
- **Memory Usage**: Large files require substantial memory for parsing

### Compatibility
- **Browser Support**: Format supported by Chrome, Firefox, Safari, Edge
- **Version Compatibility**: Format has remained stable across Chrome versions
- **Import/Export**: Standard format for bookmark transfer between browsers

## Implementation Guidelines

### Creating Bookmarks Files
1. Start with proper DOCTYPE and meta tags
2. Use proper UTF-8 encoding
3. Maintain proper nesting of DL/DT elements
4. Include paragraph tags after DL elements
5. Use Unix timestamps for dates
6. Encode special characters properly
7. Include favicon data when available

### Parsing Bookmarks Files
1. Parse as HTML document
2. Navigate DOM structure starting from root DL
3. Handle missing or malformed attributes gracefully
4. Convert timestamps to appropriate date objects
5. Decode base64 favicon data if needed
6. Preserve folder hierarchy in data structures

### Security Considerations
- **Data URLs**: Validate favicon data URLs to prevent XSS
- **File Paths**: Sanitize file:// URLs appropriately
- **Content Validation**: Validate HTML content when importing
- **Size Limits**: Implement reasonable file size limits

## Browser-Specific Notes

### Chrome Export
- Always includes favicon data when available
- Uses consistent timestamp format
- Maintains stable folder ordering
- Includes PERSONAL_TOOLBAR_FOLDER attribute
- Does not export internal folder IDs visible in the UI

### Chrome Import
- Accessible via `chrome://bookmarks/` → Organize → Import bookmarks
- Also available through `chrome://settings/importData`
- Assigns new internal folder IDs during import process
- Preserves folder hierarchy and nesting structure
- Bookmark display format combines title and URL: "Title https://url.com/"

### Import Compatibility
- Other browsers may ignore some Chrome-specific attributes
- Favicon data may be processed differently
- Folder structures generally preserved across browsers
- Timestamps may be converted to browser-native formats

### Chrome UI Features
- **Search**: Real-time bookmark search across all folders
- **Organize Menu**: Add bookmarks/folders, import/export, sort options
- **Grid View**: Shows bookmark title, URL, and action buttons
- **Tree Navigation**: Hierarchical folder structure with expand/collapse
- **Drag & Drop**: Bookmarks and folders can be reorganized (not reflected in export format)

## Future Considerations

### Format Evolution
- **JSON Alternative**: Chrome also supports JSON format internally
- **Extended Metadata**: Future versions may include additional attributes
- **Performance Optimizations**: Format may evolve for better performance
- **Security Enhancements**: Additional validation may be added

This specification provides a comprehensive foundation for working with Chrome bookmark files in future coding projects, enabling reliable parsing, generation, and manipulation of bookmark data.