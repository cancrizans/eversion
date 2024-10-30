in vec3 pos;
in vec3 nor;
// in vec2 uvs;
uniform vec4 color;
uniform vec3 cameraPosition;

layout (location = 0) out vec4 outColor;

void main()
{

    vec3 normal = normalize(gl_FrontFacing ? nor : -nor);

    vec3 view = normalize(cameraPosition - pos);



    // vec2 offcheck = fract(uvs * 8.0  / 3.14) - 0.5;

    // if (offcheck.x > 0){
    //     discard;
    // }

    float fresnel = clamp(1.0 - dot(normal,view),0.0,1.0);

    vec3 ocol = mix(
        color.rgb * (normal.y + 1.0) / 2.0,
        vec3(1.0,1.0,1.0),
        fresnel * 0.5
    );


    outColor = vec4(ocol,1.0);
}